### Description
<!-- Describe your changes. -->



### Motivation and Context
<!-- - Why is this change required? What problem does it solve?
- If it fixes an open issue, please link to the issue here. -->

### Fork PR device coverage
Fork PRs get Tier 1 CI automatically (builds, lint, x86_64 tests, coverage, ASan). Device, QDC and wheel-smoke tests need internal credentials, so on a fork they report a green placeholder and the PR stays mergeable; that coverage runs post-merge on main, or before merge when a maintainer comments `/ci`. See [CONTRIBUTING.md](../CONTRIBUTING.md#ci-for-fork-pull-requests).
<!-- Maintainer: after `/ci`, paste the run link here. -->


