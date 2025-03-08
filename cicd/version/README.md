# GitOps version

- Generate a version from git tags and commits
- Version is cached in a file to avoid expensive git operations
- Schema is based on [semver](https://semver.org/)
- Compatible with [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
- Portable, works on Linux and Windows

## In a nutshell

Version are generated from git tags and commits. Format is "[major].{minor}.[patch]-[commit amout since last tag].[commit id]+[timestamp]".

Timestamp is only added if the version is not cached. Timestamp is added to avoid integrity issues when executing the script multiple times.

### Install to your repo

Linux:

```bash
# Add this repo as a submodule, from the root of your repo
❯ git submodule add -b master https://github.com/clemlesne/gitops-version cicd/version
```

In GitHub, [Dependabot](https://github.com/dependabot) can automatically create you pull requests for new versions:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: gitsubmodule
    schedule:
        interval: daily
    directory: cicd/version
```

### How to use

```bash
❯ sh version.sh -g .
0.2.11-44.630dcd2
```

```powershell
❯ .\version.ps1 -g .
0.2.11-44.630dcd2
```

Examples:

```bash
# Get short version from current commit
❯ sh version.sh -g .
0.2.11-44.630dcd2

# Get long version from current commit
❯ sh version.sh -g . -m
0.2.11-44.630dcd2+20230327090732

# Get the long cached version from commit
❯ sh version.sh -g . -m -c
0.2.11-44.630dcd2+20230327090732

# Get the version from a repo stored in another folder
❯ sh version.sh -g my_folder/
0.7.2
```

## Advanced usage

### [Makefile](https://www.gnu.org/software/make/manual/make.html)

In your `Makefile`:

<!-- markdownlint-disable no-hard-tabs -->
```makefile
version:
	@bash cicd/version/version.sh -g . -c

version-full:
	@bash cicd/version/version.sh -g . -c -m
```
<!-- markdownlint-enable no-hard-tabs -->

And then, in your CI:

```bash
❯ make version-full
0.7.3-21.00736a8+20230327092242
```

### [Gradle](https://gradle.org) (Java, Kotlin, Groovy, Scala, C/C++, JavaScript)

In your `build.gradle`:

```groovy
if (project.hasProperty('projVersion')) {
  project.version = project.projVersion
} else {
  project.version = 'unknown'
}
```

And then, in your CI:

```bash
# Build a Gradle project with the version
❯ gradle -PprojVersion=$(sh version.sh -g . -m -c) build
```

## [Authors](./AUTHORS.md)
