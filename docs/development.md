# Check a change before merge

[Documentation](README.md) · [Main guide](../README.md)

Use Python 3.11 or later. Run these commands from the project folder.

## Install test tools

Use a separate Python environment for development.

```bash
python -m venv .venv
```

On macOS or Linux, activate the environment:

```bash
source .venv/bin/activate
```

On Windows, use PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package and test tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev,transformers]' build twine
```

The `transformers` extra is necessary for tests that use all model types.

## Run the tests

```bash
python -m pytest -q -rs
```

The test report lists each skipped test and its cause.
CUDA and MPS tests are skipped when the required hardware is not available.
Review skipped tests before you report test results.

The `pr_merge.yml` workflow runs all tests on Python 3.11 and 3.12 with the CPU.
It runs when a pull request opens, receives new commits, or reopens.
It also runs for merge queues and pushes to `develop`.
These checks do not require a version change.
The Python 3.11 job also checks the package files.

## Require tests before merge

The workflow runs the tests. A GitHub ruleset controls whether a merge is permitted.
The workflow filename does not enforce a merge rule.
See [GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets).

The repository uses the [PR merge ruleset](https://github.com/akhanal47/bicbioseg/rules/12119415).
It applies to the default branch and branches named `release-*`.
It requires these GitHub Actions checks:

- `Python 3.11 CPU tests`
- `Python 3.12 CPU tests`

The pull request must include the latest target branch changes before merge.
The existing administrator bypass remains available to the repository owner.
Tests still run on the owner's pull requests.
The bypass permits a merge without passing checks.

Push the workflow file before contributors need these checks.
If you rename a required job, update its name in the ruleset.

## Check the package files

Use a new output folder for each build. This prevents old files from entering the check.

```bash
python -m build --outdir release-check
python -m twine check --strict release-check/*
```

These commands create and check a source distribution and a wheel.
They do not publish a release.

## Report a problem

Include the command that failed, the error message, and the package version.
Include the model settings, image size, image type, and mask labels when these details apply.
Use a small example that can reproduce the problem.

```python
from importlib.metadata import version
from bicbioseg import environment_info

print(version("bicbioseg"))
print(environment_info())
```

## Write comments and instructions

Start each code comment with `#`. Keep each comment on one line.
Use ASD-STE100 Simplified Technical English for new or changed documentation.
Use short sentences. Give one instruction per sentence.
Keep API names and parameter values exact.
