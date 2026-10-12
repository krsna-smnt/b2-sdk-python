######################################################################
#
# File: test/unit/test_stale_bucket_id_policy.py
#
# Copyright 2026 Backblaze Inc. All Rights Reserved.
#
# License https://www.backblaze.com/using_b2_code.html
#
######################################################################
"""
B2_Command_Line_Tool#567: a bucket deleted (and maybe recreated under the same name) leaves a stale
name->id cache entry. B2Api(stale_bucket_id_policy='raise'|'evict'|'heal') decides what happens when
the server rejects such an id. Every test counts server calls: the default must be byte-for-byte
legacy behaviour, and no policy may cost more calls than the legacy recovery path.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

import pytest
from apiver_deps import AuthInfoCache, B2Api, B2HttpApiConfig, RawSimulator, SqliteAccountInfo
from apiver_deps_exception import BucketIdNotFound  # the class THIS apiver's users catch

from b2sdk._internal import raw_simulator as internal_simulator

COUNTED = ('list_buckets', 'get_upload_url', 'list_file_names', 'delete_bucket', 'update_bucket')


@pytest.fixture(autouse=True)
def server_like_simulator(monkeypatch):
    """Like the real server (400 bad_bucket_id), raise this apiver's BucketIdNotFound."""

    def get_bucket_by_id(self, bucket_id):
        if bucket_id not in self.bucket_id_to_bucket:
            raise BucketIdNotFound(bucket_id)
        return self.bucket_id_to_bucket[bucket_id]

    monkeypatch.setattr(internal_simulator.RawSimulator, '_get_bucket_by_id', get_bucket_by_id)


class Cloud:
    def __init__(self):
        self.sim = RawSimulator(b2_http=None)
        self.key = self.sim.create_account()
        self.calls = Counter()
        for name in COUNTED:
            original = getattr(self.sim, name)

            def counted(*a, __n=name, __o=original, **k):
                self.calls[__n] += 1
                return __o(*a, **k)

            setattr(self.sim, name, counted)

    def api(self, path, policy=None, authorize=True):
        info = SqliteAccountInfo(file_name=str(path))
        kwargs = {} if policy is None else {'stale_bucket_id_policy': policy}
        api = B2Api(
            info,
            cache=AuthInfoCache(info),
            api_config=B2HttpApiConfig(_raw_api_class=lambda *a, **k: self.sim),
        )
        if policy is not None:  # legacy apivers don't expose the kwarg; set it on the session
            api.session.stale_bucket_id_policy = policy
        if authorize:
            api.authorize_account(
                application_key_id=self.key[0], application_key=self.key[1], realm='production'
            )
        return api


def rows(path):
    with sqlite3.connect(str(path)) as conn:
        return sorted(conn.execute('SELECT bucket_name, bucket_id FROM bucket').fetchall())


@pytest.fixture
def stale(tmp_path):
    """laptop cached foo -> old id; another machine deleted and recreated foo -> new id."""
    cloud = Cloud()
    laptop = cloud.api(tmp_path / 'laptop.sqlite')
    other = cloud.api(tmp_path / 'other.sqlite')
    old = laptop.create_bucket('foo', 'allPrivate')
    other.delete_bucket(other.get_bucket_by_name('foo'))
    new = other.create_bucket('foo', 'allPrivate')
    cloud.calls.clear()
    return cloud, tmp_path / 'laptop.sqlite', old.id_, new.id_


def next_run(cloud, path, policy):
    """The next CLI invocation: stored credentials and cache, no re-authorize."""
    return cloud.api(path, policy=policy, authorize=False)


@pytest.mark.parametrize('policy', [None, 'raise'])
def test_default_is_legacy_behaviour(stale, policy):
    cloud, path, old_id, _ = stale
    api = next_run(cloud, path, policy)
    for _ in range(2):  # legacy: fails every time, cache never changes, no extra calls
        with pytest.raises(BucketIdNotFound) as exc_info:
            list(api.get_bucket_by_name('foo').ls())
        assert type(exc_info.value) is BucketIdNotFound
        assert not hasattr(exc_info.value, 'stale_bucket_name')
    assert rows(path) == [('foo', old_id)]
    assert cloud.calls == Counter({'list_file_names': 2})


def test_evict_raises_original_exception_and_next_run_works(stale):
    cloud, path, _, new_id = stale
    api = next_run(cloud, path, 'evict')
    with pytest.raises(BucketIdNotFound) as exc_info:
        list(api.get_bucket_by_name('foo').ls())
    assert type(exc_info.value) is BucketIdNotFound  # same class this apiver always raised
    assert exc_info.value.stale_bucket_name == 'foo'
    assert rows(path) == []  # stale entry gone, no API call spent on it
    assert cloud.calls == Counter({'list_file_names': 1})

    list(next_run(cloud, path, 'evict').get_bucket_by_name('foo').ls())  # re-run
    assert rows(path) == [('foo', new_id)]
    # the re-run's one lookup replaces the legacy workaround's `b2 bucket list`
    assert cloud.calls == Counter({'list_file_names': 2, 'list_buckets': 1})


def test_heal_retries_once_with_one_lookup(stale):
    cloud, path, _, new_id = stale
    api = next_run(cloud, path, 'heal')
    bucket = api.get_bucket_by_name('foo')
    assert list(bucket.ls()) == []
    assert list(bucket.ls()) == []  # same stale Bucket object: remapped, no second lookup
    assert rows(path) == [('foo', new_id)]
    assert cloud.calls == Counter({'list_file_names': 3, 'list_buckets': 1})


def test_heal_gone_for_good_raises_original(tmp_path):
    cloud = Cloud()
    laptop = cloud.api(tmp_path / 'l.sqlite')
    other = cloud.api(tmp_path / 'o.sqlite')
    laptop.create_bucket('foo', 'allPrivate')
    other.delete_bucket(other.get_bucket_by_name('foo'))
    cloud.calls.clear()
    with pytest.raises(BucketIdNotFound) as exc_info:
        list(next_run(cloud, tmp_path / 'l.sqlite', 'heal').get_bucket_by_name('foo').ls())
    assert type(exc_info.value) is BucketIdNotFound and exc_info.value.stale_bucket_name == 'foo'
    assert rows(tmp_path / 'l.sqlite') == []
    assert cloud.calls == Counter({'list_file_names': 1, 'list_buckets': 1})


@pytest.mark.parametrize('policy', ['evict', 'heal'])
def test_delete_is_never_retargeted(stale, policy):
    cloud, path, _, new_id = stale
    api = next_run(cloud, path, policy)
    with pytest.raises(BucketIdNotFound):
        api.delete_bucket(api.get_bucket_by_name('foo'))
    assert new_id in cloud.sim.bucket_id_to_bucket  # the recreated bucket survives
    assert cloud.calls == Counter({'delete_bucket': 1})


@pytest.mark.parametrize('policy', ['raise', 'evict', 'heal'])
def test_normal_path_identical(tmp_path, policy):
    cloud = Cloud()
    api = cloud.api(tmp_path / 'n.sqlite', policy=policy)
    api.create_bucket('foo', 'allPrivate')
    cloud.calls.clear()
    bucket = next_run(cloud, tmp_path / 'n.sqlite', policy).get_bucket_by_name('foo')
    bucket.upload_bytes(b'x', 'a.txt')
    list(bucket.ls())
    assert cloud.calls == Counter({'get_upload_url': 1, 'list_file_names': 1})


def test_same_tool_delete_and_recreate_needs_nothing(tmp_path):
    """Solution A part, always on: local hygiene, zero API calls."""
    cloud = Cloud()
    api = cloud.api(tmp_path / 's.sqlite')
    api.create_bucket('foo', 'allPrivate')
    api.delete_bucket(api.get_bucket_by_name('foo'))
    new = api.create_bucket('foo', 'allPrivate')
    assert rows(tmp_path / 's.sqlite') == [('foo', new.id_)]
    cloud.calls.clear()
    list(next_run(cloud, tmp_path / 's.sqlite', None).get_bucket_by_name('foo').ls())
    assert cloud.calls == Counter({'list_file_names': 1})


def test_filtered_refresh_replaces_row(stale):
    cloud, path, _, new_id = stale
    next_run(cloud, path, None).list_buckets(bucket_name='foo')
    assert rows(path) == [('foo', new_id)]


def test_invalid_policy_rejected():
    if 'stale_bucket_id_policy' not in B2Api.__init__.__code__.co_varnames:
        pytest.skip('legacy apiver does not expose the kwarg')
    with pytest.raises(ValueError):
        B2Api(stale_bucket_id_policy='sometimes')
