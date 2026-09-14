### Description
<!-- Describe your changes. -->



### Motivation and Context
<!-- - Why is this change required? What problem does it solve?
- If it fixes an open issue, please link to the issue here. -->

### Fork PR device coverage
Fork PRs automatically get Tier 1 CI (build, x86_64 tests, lint, coverage, ASan). Device/QDC/wheel-smoke tests need internal secrets and report a green placeholder on forks; the full suite runs post-merge on main, or on demand when a maintainer comments `/ci`. See CONTRIBUTING.md ("CI for fork pull requests").
<!-- Maintainer: after `/ci`, paste the Actions run link here. -->


