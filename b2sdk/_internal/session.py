######################################################################
#
# File: b2sdk/_internal/session.py
#
# Copyright 2019 Backblaze Inc. All Rights Reserved.
#
# License https://www.backblaze.com/using_b2_code.html
#
######################################################################
from __future__ import annotations

import logging
import threading
from enum import Enum, unique
from functools import partial
from typing import Any

from b2sdk._internal.account_info.abstract import AbstractAccountInfo
from b2sdk._internal.account_info.exception import MissingAccountData
from b2sdk._internal.account_info.sqlite_account_info import SqliteAccountInfo
from b2sdk._internal.api_config import DEFAULT_HTTP_API_CONFIG, B2HttpApiConfig
from b2sdk._internal.b2http import B2Http
from b2sdk._internal.cache import AbstractCache, AuthInfoCache, DummyCache
from b2sdk._internal.encryption.setting import EncryptionSetting
from b2sdk._internal.exception import B2Error, BucketIdNotFound, InvalidAuthToken, Unauthorized
from b2sdk._internal.file_lock import BucketRetentionSetting, FileRetentionSetting, LegalHold
from b2sdk._internal.raw_api import ALL_CAPABILITIES, REALM_URLS, LifecycleRule
from b2sdk._internal.replication.setting import ReplicationConfiguration

logger = logging.getLogger(__name__)

# What to do when the server rejects a bucket id that came from the local name->id cache (#567)
STALE_BUCKET_ID_POLICIES = ('raise', 'evict', 'heal')


class _NameId:
    """Minimal (name, id_) pair accepted by AbstractCache.save_bucket."""

    __slots__ = ('name', 'id_')

    def __init__(self, name, id_):
        self.name, self.id_ = name, id_


@unique
class TokenType(Enum):
    API = 'api'
    API_TOKEN_ONLY = 'api_token_only'
    UPLOAD_PART = 'upload_part'
    UPLOAD_SMALL = 'upload_small'


class B2Session:
    """
    A facade that supplies the correct api_url and account_auth_token
    to methods of underlying raw_api and reauthorizes if necessary.
    """

    SQLITE_ACCOUNT_INFO_CLASS = staticmethod(SqliteAccountInfo)
    B2HTTP_CLASS = staticmethod(B2Http)

    def __init__(
        self,
        account_info: AbstractAccountInfo | None = None,
        cache: AbstractCache | None = None,
        api_config: B2HttpApiConfig = DEFAULT_HTTP_API_CONFIG,
    ):
        """
        Initialize Session using given account info.

        :param account_info: an instance of :class:`~b2sdk.v2.UrlPoolAccountInfo`,
                      or any custom class derived from
                      :class:`~b2sdk.v2.AbstractAccountInfo`
                      To learn more about Account Info objects, see here
                      :class:`~b2sdk.v2.SqliteAccountInfo`

        :param cache: an instance of the one of the following classes:
                      :class:`~b2sdk._internal.cache.DummyCache`, :class:`~b2sdk._internal.cache.InMemoryCache`,
                      :class:`~b2sdk._internal.cache.AuthInfoCache`,
                      or any custom class derived from :class:`~b2sdk._internal.cache.AbstractCache`
                      It is used by B2Api to cache the mapping between bucket name and bucket ids.
                      default is :class:`~b2sdk._internal.cache.DummyCache`

        :param api_config
        """

        self.raw_api = api_config.raw_api_class(self.B2HTTP_CLASS(api_config))
        if account_info is None:
            account_info = self.SQLITE_ACCOUNT_INFO_CLASS()
        if cache is None:
            if account_info:
                cache = AuthInfoCache(account_info)
            else:
                cache = DummyCache()

        self.account_info = account_info
        self.cache = cache
        self._token_callbacks = {
            TokenType.API: self._api_token_callback,
            TokenType.API_TOKEN_ONLY: self._api_token_only_callback,
            TokenType.UPLOAD_SMALL: self._upload_small,
            TokenType.UPLOAD_PART: self._upload_part,
        }
        # 'raise' (default) = legacy behaviour, untouched. See B2Api(stale_bucket_id_policy=...).
        self.stale_bucket_id_policy = 'raise'
        self._bucket_id_remap: dict[str, str] = {}  # stale id -> current id, only after a heal
        self._heal_lock = threading.Lock()
        self._no_heal = threading.local()  # set by delete_bucket; read only on the error path

    def authorize_automatically(self):
        """
        Perform automatic account authorization, retrieving all account data
        from account info object passed during initialization.
        """
        try:
            self.authorize_account(
                self.account_info.get_realm(),
                self.account_info.get_application_key_id(),
                self.account_info.get_application_key(),
            )
        except MissingAccountData:
            return False
        return True

    def authorize_account(self, realm, application_key_id, application_key):
        """
        Perform account authorization.

        :param str realm: a realm to authorize account in (usually just "production")
        :param str application_key_id: :term:`application key ID`
        :param str application_key: user's :term:`application key`
        """
        # Authorize
        realm_url = REALM_URLS.get(realm, realm)
        response = self.raw_api.authorize_account(realm_url, application_key_id, application_key)
        account_id = response['accountId']
        storage_api_info = response['apiInfo']['storageApi']

        allowed = self._construct_allowed_dict(storage_api_info)

        # Clear the cache if new account has been used
        if not self.account_info.is_same_account(account_id, realm):
            self.cache.clear()

        # Store the auth data
        self.account_info.set_auth_data(
            account_id=account_id,
            auth_token=response['authorizationToken'],
            api_url=storage_api_info['apiUrl'],
            download_url=storage_api_info['downloadUrl'],
            absolute_minimum_part_size=storage_api_info['absoluteMinimumPartSize'],
            recommended_part_size=storage_api_info['recommendedPartSize'],
            application_key=application_key,
            realm=realm,
            s3_api_url=storage_api_info['s3ApiUrl'],
            allowed=allowed,
            application_key_id=application_key_id,
        )

    def _construct_allowed_dict(self, storage_api_info):
        # `allowed` object has been deprecated in the v3 of the API, but we still
        # construct it artificially to avoid changes in all the reliant parts.

        return storage_api_info['allowed']

    def cancel_large_file(self, file_id):
        return self._wrap_default_token(self.raw_api.cancel_large_file, file_id)

    def create_bucket(
        self,
        account_id,
        bucket_name,
        bucket_type,
        bucket_info=None,
        cors_rules=None,
        lifecycle_rules: list[LifecycleRule] | None = None,
        default_server_side_encryption=None,
        is_file_lock_enabled: bool | None = None,
        replication: ReplicationConfiguration | None = None,
    ):
        return self._wrap_default_token(
            self.raw_api.create_bucket,
            account_id,
            bucket_name,
            bucket_type,
            bucket_info=bucket_info,
            cors_rules=cors_rules,
            lifecycle_rules=lifecycle_rules,
            default_server_side_encryption=default_server_side_encryption,
            is_file_lock_enabled=is_file_lock_enabled,
            replication=replication,
        )

    def create_key(
        self, account_id, capabilities, key_name, valid_duration_seconds, bucket_ids, name_prefix
    ):
        return self._wrap_default_token(
            self.raw_api.create_key,
            account_id,
            capabilities,
            key_name,
            valid_duration_seconds,
            bucket_ids,
            name_prefix,
        )

    def delete_key(self, application_key_id):
        return self._wrap_default_token(self.raw_api.delete_key, application_key_id)

    def delete_bucket(self, account_id, bucket_id):
        # A stale id must never be retargeted to a *different* bucket that now holds the name, so
        # healing is switched off for this call. A flag (not a method comparison) so wrappers
        # cannot defeat it; it is only read on the error path, so the normal path is unchanged.
        self._no_heal.active = True
        try:
            response = self._wrap_default_token(self.raw_api.delete_bucket, account_id, bucket_id)
        finally:
            self._no_heal.active = False
        # Forget the deleted bucket's name so a later recreate is resolved fresh (#567). Done here,
        # not in B2Api, so every apiver gets it (v0 overrides B2Api.delete_bucket). Local write only.
        name = self.cache.get_bucket_name_or_none_from_bucket_id(bucket_id)
        if name is not None:
            self.cache.remove_bucket_name(name)
        return response

    def delete_file_version(self, file_id, file_name, bypass_governance: bool = False):
        return self._wrap_default_token(
            self.raw_api.delete_file_version, file_id, file_name, bypass_governance
        )

    def download_file_from_url(
        self,
        url: str,
        range_: tuple[int, int] | None = None,
        encryption: EncryptionSetting | None = None,
    ):
        return self._wrap_token(
            self.raw_api.download_file_from_url,
            TokenType.API_TOKEN_ONLY,
            url,
            range_=range_,
            encryption=encryption,
        )

    def finish_large_file(self, file_id, part_sha1_array):
        return self._wrap_default_token(self.raw_api.finish_large_file, file_id, part_sha1_array)

    def get_download_authorization(self, bucket_id, file_name_prefix, valid_duration_in_seconds):
        return self._wrap_default_token(
            self.raw_api.get_download_authorization,
            bucket_id,
            file_name_prefix,
            valid_duration_in_seconds,
        )

    def get_file_info_by_id(self, file_id: str) -> dict[str, Any]:
        return self._wrap_default_token(self.raw_api.get_file_info_by_id, file_id)

    def get_file_info_by_name(self, bucket_name: str, file_name: str) -> dict[str, Any]:
        return self._wrap_default_token(self.raw_api.get_file_info_by_name, bucket_name, file_name)

    def get_upload_url(self, bucket_id):
        return self._wrap_default_token(self.raw_api.get_upload_url, bucket_id)

    def get_upload_part_url(self, file_id):
        return self._wrap_default_token(self.raw_api.get_upload_part_url, file_id)

    def hide_file(self, bucket_id, file_name):
        return self._wrap_default_token(self.raw_api.hide_file, bucket_id, file_name)

    def list_buckets(self, account_id, bucket_id=None, bucket_name=None):
        return self._wrap_default_token(
            self.raw_api.list_buckets,
            account_id,
            bucket_id=bucket_id,
            bucket_name=bucket_name,
        )

    def list_file_names(
        self,
        bucket_id,
        start_file_name=None,
        max_file_count=None,
        prefix=None,
    ):
        return self._wrap_default_token(
            self.raw_api.list_file_names,
            bucket_id,
            start_file_name=start_file_name,
            max_file_count=max_file_count,
            prefix=prefix,
        )

    def list_file_versions(
        self,
        bucket_id,
        start_file_name=None,
        start_file_id=None,
        max_file_count=None,
        prefix=None,
    ):
        return self._wrap_default_token(
            self.raw_api.list_file_versions,
            bucket_id,
            start_file_name=start_file_name,
            start_file_id=start_file_id,
            max_file_count=max_file_count,
            prefix=prefix,
        )

    def list_keys(self, account_id, max_key_count=None, start_application_key_id=None):
        return self._wrap_default_token(
            self.raw_api.list_keys,
            account_id,
            max_key_count=max_key_count,
            start_application_key_id=start_application_key_id,
        )

    def list_parts(self, file_id, start_part_number, max_part_count):
        return self._wrap_default_token(
            self.raw_api.list_parts, file_id, start_part_number, max_part_count
        )

    def list_unfinished_large_files(
        self,
        bucket_id,
        start_file_id=None,
        max_file_count=None,
        prefix=None,
    ):
        return self._wrap_default_token(
            self.raw_api.list_unfinished_large_files,
            bucket_id,
            start_file_id=start_file_id,
            max_file_count=max_file_count,
            prefix=prefix,
        )

    def start_large_file(
        self,
        bucket_id,
        file_name,
        content_type,
        file_info,
        server_side_encryption: EncryptionSetting | None = None,
        file_retention: FileRetentionSetting | None = None,
        legal_hold: LegalHold | None = None,
        custom_upload_timestamp: int | None = None,
    ):
        return self._wrap_default_token(
            self.raw_api.start_large_file,
            bucket_id,
            file_name,
            content_type,
            file_info,
            server_side_encryption,
            file_retention=file_retention,
            legal_hold=legal_hold,
            custom_upload_timestamp=custom_upload_timestamp,
        )

    def update_bucket(
        self,
        account_id,
        bucket_id,
        bucket_type=None,
        bucket_info=None,
        cors_rules=None,
        lifecycle_rules: list[LifecycleRule] | None = None,
        if_revision_is=None,
        default_server_side_encryption: EncryptionSetting | None = None,
        default_retention: BucketRetentionSetting | None = None,
        replication: ReplicationConfiguration | None = None,
        is_file_lock_enabled: bool | None = None,
    ):
        return self._wrap_default_token(
            self.raw_api.update_bucket,
            account_id,
            bucket_id,
            bucket_type=bucket_type,
            bucket_info=bucket_info,
            cors_rules=cors_rules,
            lifecycle_rules=lifecycle_rules,
            if_revision_is=if_revision_is,
            default_server_side_encryption=default_server_side_encryption,
            default_retention=default_retention,
            replication=replication,
            is_file_lock_enabled=is_file_lock_enabled,
        )

    def upload_file(
        self,
        bucket_id,
        file_name,
        content_length,
        content_type,
        content_sha1,
        file_info,
        data_stream,
        server_side_encryption: EncryptionSetting | None = None,
        file_retention: FileRetentionSetting | None = None,
        legal_hold: LegalHold | None = None,
        custom_upload_timestamp: int | None = None,
    ):
        return self._wrap_token(
            self.raw_api.upload_file,
            TokenType.UPLOAD_SMALL,
            bucket_id,
            file_name,
            content_length,
            content_type,
            content_sha1,
            file_info,
            data_stream,
            server_side_encryption,
            file_retention=file_retention,
            legal_hold=legal_hold,
            custom_upload_timestamp=custom_upload_timestamp,
        )

    def upload_part(
        self,
        file_id,
        part_number,
        content_length,
        sha1_sum,
        input_stream,
        server_side_encryption: EncryptionSetting | None = None,
    ):
        return self._wrap_token(
            self.raw_api.upload_part,
            TokenType.UPLOAD_PART,
            file_id,
            part_number,
            content_length,
            sha1_sum,
            input_stream,
            server_side_encryption,
        )

    def get_download_url_by_id(self, file_id):
        return self.raw_api.get_download_url_by_id(self.account_info.get_download_url(), file_id)

    def get_download_url_by_name(self, bucket_name, file_name):
        return self.raw_api.get_download_url_by_name(
            self.account_info.get_download_url(), bucket_name, file_name
        )

    def copy_file(
        self,
        source_file_id,
        new_file_name,
        bytes_range=None,
        metadata_directive=None,
        content_type=None,
        file_info=None,
        destination_bucket_id=None,
        destination_server_side_encryption: EncryptionSetting | None = None,
        source_server_side_encryption: EncryptionSetting | None = None,
        file_retention: FileRetentionSetting | None = None,
        legal_hold: LegalHold | None = None,
    ):
        return self._wrap_default_token(
            self.raw_api.copy_file,
            source_file_id,
            new_file_name,
            bytes_range=bytes_range,
            metadata_directive=metadata_directive,
            content_type=content_type,
            file_info=file_info,
            destination_bucket_id=destination_bucket_id,
            destination_server_side_encryption=destination_server_side_encryption,
            source_server_side_encryption=source_server_side_encryption,
            file_retention=file_retention,
            legal_hold=legal_hold,
        )

    def copy_part(
        self,
        source_file_id,
        large_file_id,
        part_number,
        bytes_range=None,
        destination_server_side_encryption: EncryptionSetting | None = None,
        source_server_side_encryption: EncryptionSetting | None = None,
    ):
        return self._wrap_default_token(
            self.raw_api.copy_part,
            source_file_id,
            large_file_id,
            part_number,
            bytes_range=bytes_range,
            destination_server_side_encryption=destination_server_side_encryption,
            source_server_side_encryption=source_server_side_encryption,
        )

    def _wrap_default_token(self, raw_api_method, *args, **kwargs):
        return self._wrap_token(raw_api_method, TokenType.API, *args, **kwargs)

    def _wrap_token(self, raw_api_method, token_type, *args, **kwargs):
        callback = self._token_callbacks[token_type]
        partial_callback = partial(callback, raw_api_method, *args, **kwargs)

        try:
            return self._execute_with_auth_retry(partial_callback)
        except BucketIdNotFound as exc:
            # everything below runs only after the server rejected a bucket id (#567)
            if self.stale_bucket_id_policy == 'raise' or not self._prepare_retry(exc, args, kwargs):
                raise  # the ORIGINAL exception object: type and message unchanged
        return self._retry_once_after_heal(callback, raw_api_method, args, kwargs)

    def _wrap_token_remapped(self, raw_api_method, token_type, *args, **kwargs):
        """
        Installed on this session instance by the first successful heal (policy 'heal' only), so
        that stale ids still held by long-lived Bucket objects go to the current bucket. Until a
        heal happens, _wrap_token above is the only path and costs nothing extra.
        """
        if not self._healing_disabled():
            args, kwargs = self._apply_bucket_id_remap(args, kwargs)
        return type(self)._wrap_token(self, raw_api_method, token_type, *args, **kwargs)

    def _healing_disabled(self) -> bool:
        return getattr(self._no_heal, 'active', False)

    def _prepare_retry(self, exc, args, kwargs) -> bool:
        """Policy evict/heal: evict the stale entry; for heal, also find the current id."""
        name = self._evict_stale_bucket_id(exc, args, kwargs)
        if name is None or self._healing_disabled():
            return False
        return self._heal(exc, name)

    def _retry_once_after_heal(self, callback, raw_api_method, args, kwargs):
        args, kwargs = self._apply_bucket_id_remap(args, kwargs)
        try:  # exactly one retry, against the bucket that now holds the name
            return self._execute_with_auth_retry(partial(callback, raw_api_method, *args, **kwargs))
        except BucketIdNotFound as exc:
            self._evict_stale_bucket_id(exc, args, kwargs)
            raise

    def _evict_stale_bucket_id(self, exc, args, kwargs) -> str | None:
        """
        If the rejected id was one we sent and it came from the name->id cache: drop that entry
        (local write, no API call) and annotate the original exception with the bucket name.
        """
        bucket_id = getattr(exc, 'bucket_id', None)
        if not bucket_id or (bucket_id not in args and bucket_id not in kwargs.values()):
            return None
        name = self.cache.get_bucket_name_or_none_from_bucket_id(bucket_id)
        if name is None:
            return None  # not from the cache (or DummyCache): nothing to fix
        if not self._key_can_list_buckets():
            # evicting would leave a key that cannot resolve names unable to use the bucket at all
            # (Unauthorized instead of BucketIdNotFound next time): keep legacy behaviour instead
            return None
        self.cache.remove_bucket_name(name)
        exc.stale_bucket_name = name  # additive attribute; callers may use it for a better message
        exc.stale_bucket_key_bound = bucket_id in self._key_restricted_bucket_ids()
        return name

    def _heal(self, exc, name) -> bool:
        """
        policy 'heal' only: one b2_list_buckets(bucketName=name) to find the bucket that now holds
        the name. False (caller re-raises the original error) if the key itself is bound to the
        deleted bucket, the lookup fails, or the bucket is gone for good.
        """
        if self.stale_bucket_id_policy != 'heal' or exc.stale_bucket_key_bound:
            return False
        with self._heal_lock:
            if exc.bucket_id in self._bucket_id_remap:
                return True  # healed by another thread meanwhile
            try:
                response = self.list_buckets(self.account_info.get_account_id(), bucket_name=name)
            except B2Error:
                logger.debug('lookup of %r failed while healing', name, exc_info=True)
                return False
            fresh_id = next(
                (b['bucketId'] for b in response['buckets'] if b['bucketName'] == name), None
            )
            if fresh_id is None:
                return False  # deleted for good
            self.cache.save_bucket(_NameId(name, fresh_id))
            # keep the map transitive: ids that already pointed at the now-stale id follow too
            for old_id, current in list(self._bucket_id_remap.items()):
                if current == exc.bucket_id:
                    self._bucket_id_remap[old_id] = fresh_id
            self._bucket_id_remap[exc.bucket_id] = fresh_id
            self._wrap_token = self._wrap_token_remapped  # from now on, apply the remap
            logger.info('bucket %r was recreated (%s -> %s); retrying once', name, exc.bucket_id,
                        fresh_id)
            return True

    def _apply_bucket_id_remap(self, args, kwargs):
        remap = self._bucket_id_remap
        args = tuple(remap.get(a, a) if isinstance(a, str) else a for a in args)
        kwargs = {k: remap.get(v, v) if isinstance(v, str) else v for k, v in kwargs.items()}
        return args, kwargs

    def _key_can_list_buckets(self) -> bool:
        try:
            capabilities = (self.account_info.get_allowed() or {}).get('capabilities')
        except MissingAccountData:
            return False
        return capabilities is None or 'listBuckets' in capabilities

    def _key_restricted_bucket_ids(self) -> set:
        try:
            allowed = self.account_info.get_allowed() or {}
        except MissingAccountData:
            return set()
        ids = {b.get('id') for b in allowed.get('buckets') or []}
        ids.add(allowed.get('bucketId'))  # legacy single-bucket format (apiver <= v2)
        ids.discard(None)
        return ids

    def _api_token_callback(self, raw_api_method, *args, **kwargs):
        api_url = self.account_info.get_api_url()
        account_auth_token = self.account_info.get_account_auth_token()
        return raw_api_method(api_url, account_auth_token, *args, **kwargs)

    def _api_token_only_callback(self, raw_api_method, *args, **kwargs):
        account_auth_token = self.account_info.get_account_auth_token()
        return raw_api_method(account_auth_token, *args, **kwargs)

    def _execute_with_auth_retry(self, callback, first_attempt: bool = True):
        try:
            return callback()
        except InvalidAuthToken:
            if first_attempt and self.authorize_automatically():
                return self._execute_with_auth_retry(callback, first_attempt=False)
            raise
        except Unauthorized as exc:
            # When performing HEAD requests, the api can return an empty response
            # with a 401 status code, which may or may not be related to expired
            # token, thus we also try to re-authorize if explicit `unauthorized`
            # code is missing
            if not exc.code and first_attempt and self.authorize_automatically():
                return self._execute_with_auth_retry(callback, first_attempt=False)
            raise self._add_app_key_info_to_unauthorized(exc)

    def _add_app_key_info_to_unauthorized(self, unauthorized):
        """
        Take an Unauthorized error and adds information from the application key
        about why it might have failed.
        """
        # What's allowed?
        allowed = self.account_info.get_allowed()
        capabilities = allowed['capabilities']
        name_prefix = allowed['namePrefix']

        # Make a list of messages about the application key restrictions
        key_messages = []
        if set(capabilities) != set(ALL_CAPABILITIES):
            key_messages.append("with capabilities '" + ','.join(capabilities) + "'")

        allowed_buckets_msg = self._get_allowed_buckets_message(allowed)
        if allowed_buckets_msg:
            key_messages.append(allowed_buckets_msg)

        if name_prefix is not None:
            key_messages.append("restricted to files that start with '" + name_prefix + "'")
        if not key_messages:
            key_messages.append('with no restrictions')

        # Make a new message
        new_message = unauthorized.message or 'unauthorized'
        new_message += ' for application key ' + ', '.join(key_messages)

        return Unauthorized(new_message, unauthorized.code)

    def _get_allowed_buckets_message(self, allowed) -> str | None:
        buckets = allowed['buckets']
        if not buckets:
            return None

        bucket_names = [b['name'] for b in buckets]

        return f'restricted to buckets {bucket_names}'

    def _get_upload_data(self, bucket_id):
        """
        Take ownership of an upload URL / auth token for the bucket and
        return it.
        """
        account_info = self.account_info
        upload_url, upload_auth_token = account_info.take_bucket_upload_url(bucket_id)
        if None not in (upload_url, upload_auth_token):
            return upload_url, upload_auth_token

        response = self.get_upload_url(bucket_id)
        return response['uploadUrl'], response['authorizationToken']

    def _get_upload_part_data(self, file_id):
        """
        Make sure that we have an upload URL and auth token for the given bucket and
        return it.
        """
        account_info = self.account_info
        upload_url, upload_auth_token = account_info.take_large_file_upload_url(file_id)
        if None not in (upload_url, upload_auth_token):
            return upload_url, upload_auth_token

        response = self.get_upload_part_url(file_id)
        return response['uploadUrl'], response['authorizationToken']

    def _upload_small(self, f, bucket_id, *args, **kwargs):
        upload_url, upload_auth_token = self._get_upload_data(bucket_id)
        response = f(upload_url, upload_auth_token, *args, **kwargs)
        self.account_info.put_bucket_upload_url(bucket_id, upload_url, upload_auth_token)
        return response

    def _upload_part(self, f, file_id, *args, **kwargs):
        upload_url, upload_auth_token = self._get_upload_part_data(file_id)
        response = f(upload_url, upload_auth_token, *args, **kwargs)
        self.account_info.put_large_file_upload_url(file_id, upload_url, upload_auth_token)
        return response

    def update_file_retention(
        self,
        file_id,
        file_name,
        file_retention: FileRetentionSetting,
        bypass_governance: bool = False,
    ):
        return self._wrap_default_token(
            self.raw_api.update_file_retention,
            file_id,
            file_name,
            file_retention,
            bypass_governance,
        )

    def update_file_legal_hold(
        self,
        file_id,
        file_name,
        legal_hold: LegalHold,
    ):
        return self._wrap_default_token(
            self.raw_api.update_file_legal_hold,
            file_id,
            file_name,
            legal_hold,
        )

    def get_bucket_notification_rules(self, bucket_id):
        return self._wrap_default_token(self.raw_api.get_bucket_notification_rules, bucket_id)

    def set_bucket_notification_rules(self, bucket_id, rules):
        return self._wrap_default_token(
            self.raw_api.set_bucket_notification_rules, bucket_id, rules
        )
