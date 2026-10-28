# Changelog

## 0.1.1 (2026-10-28)

Release distributions on GitHub now carry detached GPG signatures produced with
Garasign; see [Verifying releases](README.md#verifying-releases) in the README
for the signing key and how to verify.

## 0.1.0 (2026-10-27)

First release of MongoDBSessionStore implementation Adapter for mirroring session transcripts to external storage.
This began as a reference example in [anthropics/claude-agent-sdk-python #1014](https://github.com/anthropics/claude-agent-sdk-python/pull/1014).
This release repurposes it as a standalone package and hardens it.

Security improvements:

- Every `SessionStore` method now validates its key before any query.
  `project_key`, `session_id` and `subpath` must be strings (`TypeError`
  otherwise) and `project_key` and `session_id` must be non-empty
  (`ValueError`). Previously a non-string key field such as `{"$ne": ""}`
  was passed into the MongoDB filter verbatim, where it acted as a query
  operator and could read or delete entries of other sessions or tenants.
  The SDK's own code paths were not affected; direct callers of the store were.
- The README gains a Security section covering the sensitivity of stored
  transcripts, TLS and authentication, least-privilege database users,
  encryption at rest, and the cross-tenant scope of `delete_inactive()`.
