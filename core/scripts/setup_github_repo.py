#!/usr/bin/env python3
"""
GitHub Repository Setup Script
自动设置GitHub仓库的完整文件结构
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class GitHubSetup:
    """GitHub Repository Setup Manager"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.github_dir = project_root / ".github"
        
    def create_issue_templates(self) -> None:
        """Create comprehensive issue templates"""
        templates_dir = self.github_dir / "ISSUE_TEMPLATE"
        templates_dir.mkdir(exist_ok=True)
        
        # Config.yml for issue templates
        config_yml = """blank_issues_enabled: false
contact_links:
  - name: 💬 Community Support
    url: https://github.com/your-repo/brain-inspired-ai/discussions
    about: Ask questions and get help from the community
  - name: 📖 Documentation
    url: https://brain-ai-docs.org
    about: Check our documentation for guides and examples
  - name: 🔍 Search Issues
    url: https://github.com/your-repo/brain-inspired-ai/issues
    about: Search for existing issues to avoid duplicates
"""
        
        (templates_dir / "config.yml").write_text(config_yml)
        
        # Question template
        question_template = """---
name: ❓ Question
about: Ask a question about the project
title: '[QUESTION] '
labels: ['question', 'needs-triage']
assignees: ''
---

## ❓ Question

Please describe your question clearly and provide context.

## 🔍 What I Tried

Please describe what you have tried so far.

## 📖 Documentation Checked

- [ ] Installation guide
- [ ] Quick start guide
- [ ] API documentation
- [ ] Examples
- [ ] GitHub issues

## 💻 Environment Details

**OS:** <!-- e.g. Ubuntu 20.04 -->
**Python Version:** <!-- e.g. 3.9.7 -->
**Brain-Inspired AI Version:** <!-- e.g. 2.0.0 -->
**Installation Method:** <!-- e.g. pip, conda, docker -->

## 📋 Additional Context

Add any other context about your question.
"""
        
        (templates_dir / "question.md").write_text(question_template)
        
    def create_security_policy(self) -> None:
        """Create security policy"""
        security_policy = """# Security Policy

## 🛡️ Security

We take the security of Brain-Inspired AI seriously. If you believe you have found a security vulnerability, please report it to us as described below.

## 🔒 Reporting Security Vulnerabilities

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them by:

1. **Email**: security@brain-ai.org
2. **Private Vulnerability Report**: Use GitHub's private vulnerability reporting feature
3. **Encrypted Email**: Use our PGP key (if available)

### What to Include

When reporting, please include:

- Type of vulnerability (buffer overflow, SQL injection, etc.)
- Full paths of source file(s) related to the manifestation of the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Process

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
2. **Investigation**: We will investigate and confirm the vulnerability
3. **Fix Development**: We will develop and test a fix
4. **Release**: We will release the fix and notify users
5. **Credit**: We will credit you for the discovery (unless you prefer to remain anonymous)

## 🔍 Security Measures

### Code Review
- All code changes go through security review
- Automated security scanning with every PR
- Regular dependency vulnerability scanning

### Dependencies
- Regular updates of all dependencies
- Automated security updates for dependencies
- Strict versioning for security-critical packages

### Infrastructure
- Secure hosting and deployment
- Regular security audits
- Encrypted communication

## 📋 Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## 🎯 Responsible Disclosure

We appreciate the work of security researchers who help us keep our users safe. We follow responsible disclosure principles:

- We will not take legal action against researchers who comply with this policy
- We will acknowledge receipt of vulnerability reports within 48 hours
- We will provide regular updates on our progress
- We will publicly acknowledge researchers (with their permission)

## 📞 Contact

For security-related questions or to report vulnerabilities:

- **Security Email**: security@brain-ai.org
- **Main Repository**: https://github.com/your-repo/brain-inspired-ai
- **Documentation**: https://brain-ai-docs.org

## 🏆 Recognition

We maintain a [Security Hall of Fame](SECURITY.md) to recognize researchers who have responsibly disclosed vulnerabilities.

---

**Thank you for helping keep Brain-Inspired AI and our users safe! 🛡️**
"""
        
        (self.project_root / "SECURITY.md").write_text(security_policy)
        
    def create_funding_yml(self) -> None:
        """Create GitHub funding configuration"""
        funding_yml = """github:
  - your-username
patreon: your-patreon-username
open_collective: brain-ai
ko_fi: your-ko-fi-username
tidelift: your-tidelift-username
community_bridge: brain-ai
issuehunt:
  - your-username
lfx_crowdfunding:
  - your-username
custom:
  - name: "Brain AI Research Fund"
    url: "https://opencollective.com/brain-ai"
  - name: "University Partnership Program"
    url: "https://brain-ai.org/partnerships"
"""
        
        (self.project_root / ".github" / "FUNDING.yml").write_text(funding_yml)
        
    def create_dependabot_config(self) -> None:
        """Create Dependabot configuration"""
        dependabot_yml = """version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    commit-message:
      prefix: "chore"
      prefix-development: "chore"
      include: "scope"
    reviewers:
      - "maintainer-1"
      - "maintainer-2"
    assignees:
      - "maintainer-1"
    open-pull-requests-limit: 10
    rebase-strategy: "auto"
    
  # GitHub Actions dependencies
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    commit-message:
      prefix: "ci"
      include: "scope"
    reviewers:
      - "maintainer-1"
    assignees:
      - "maintainer-1"
    open-pull-requests-limit: 5
    
  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    commit-message:
      prefix: "docker"
      include: "scope"
    reviewers:
      - "maintainer-1"
    assignees:
      - "maintainer-1"
    open-pull-requests-limit: 5
    
  # npm dependencies (for documentation)
  - package-ecosystem: "npm"
    directory: "/docs"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    commit-message:
      prefix: "docs"
      include: "scope"
    reviewers:
      - "docs-maintainer"
    open-pull-requests-limit: 3
"""
        
        (self.github_dir / "dependabot.yml").write_text(dependabot_yml)
        
    def create_codeql_config(self) -> None:
        """Create CodeQL configuration for security analysis"""
        codeql_yml = """# For most projects, this workflow file will not need changing; you only need to modify it if you
# have changed the target language, or if you have changed the query or test suite.
#
# ******** OPTIONAL ********
# queries: Comma-separated list of additional queries to run as part of the analysis
# queries: ./path/to/local/query.ql, # other-internal query
# queries: query/security_extended.ql@security-extended
# queries: ./path/to/imported/query.ql
# 
# Language-specific queries are imported automatically for:
# - javascript and typescript
# - python and go
# - java and kotlin
# - ruby
#
# Use the following options to modify the languages analyzed:
# languages:
# - javascript
# - typescript
# - python
# - go
# - java
# - kotlin
# - ruby

name: "CodeQL Security Scan"

on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main", "develop" ]
  schedule:
    - cron: 'weekly'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]
        # CodeQL supports [ 'cpp', 'csharp', 'go', 'java', 'javascript', 'python', 'ruby' ]
        # Use only 'java' to analyze code written in Java, Kotlin or both
        # Use only 'javascript' to analyze code written in JavaScript, TypeScript or both
        # Learn more about CodeQL language support at https://aka.ms/codeql-docs/language-support

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{matrix.language}}"
"""
        
        (self.github_dir / "codeql.yml").write_text(codeql_yml)
        
    def create_stale_issues_config(self) -> None:
        """Create configuration for stale issue management"""
        stale_yml = """name: "Stale Issues and PRs"

on:
  schedule:
    - cron: "0 0 * * *"  # Daily at midnight
  workflow_dispatch:

permissions:
  issues: write
  pull-requests: write

jobs:
  stale:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/stale@v8
      with:
        repo-token: ${{ secrets.GITHUB_TOKEN }}
        
        # Issue settings
        stale-issue-message: |
          👋 Hi there! 

          Thank you for reporting this issue, but it seems to have been inactive for a while. 

          If you are still experiencing this problem, or if you have additional information that could help us reproduce it, please feel free to update this issue!

          Otherwise, we may need to close this issue due to inactivity. However, we always appreciate hearing about issues, so if you still want to keep this open, just let us know! 🙏

          Thank you for your understanding!
          - The Brain AI Team

        stale-pr-message: |
          👋 Hi there!

          Thank you for submitting this pull request! However, it seems to have been inactive for a while.

          If you have any additional changes or updates you'd like to make, please feel free to update the pull request!

          Otherwise, we may need to close this pull request due to inactivity. However, we always appreciate contributions, so if you still want to keep this open, just let us know! 🙏

          Thank you for your understanding!
          - The Brain AI Team

        stale-issue-label: "stale"
        stale-pr-label: "stale"
        
        # Timing settings
        days-before-issue-stale: 60
        days-before-pr-stale: 60
        days-before-issue-close: 30
        days-before-pr-close: 30
        
        # Exemptions
        exempt-issue-labels: "bug,enhancement,good first issue,help wanted,question,high priority"
        exempt-pr-labels: "enhancement,good first issue,high priority"
        
        # Operations
        only-labels: "needs-triage,bug,enhancement"
        operations-per-run: 100
"""
        
        (self.github_dir / "stale.yml").write_text(stale_yml)
        
    def create_auto_merge_config(self) -> None:
        """Create auto-merge configuration for dependabot PRs"""
        auto_merge_yml = """name: Auto-merge Dependabot PRs

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
  pull_request_target:
    types: [opened, synchronize, reopened, ready_for_review]

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: contains(github.event.pull_request.title, 'chore(deps)') || contains(github.event.pull_request.title, 'ci(')
    steps:
    - name: Auto-merge Dependabot PRs
      uses: pascalgn/merge-action@v0.15.6
      if: github.actor == 'dependabot[bot]' || github.actor == 'dependabot-preview[bot]'
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        merge_method: squash
        merge_commit_message: pull-request-title
        merge_commit_description: pull-request-description
        merge_commit_author: Dependabot
        delete_branch: true
"""
        
        (self.github_dir / "auto-merge.yml").write_text(auto_merge_yml)
        
    def create_branch_protection_config(self) -> None:
        """Create branch protection configuration"""
        protection_config = """# Branch Protection Rules Configuration
# This file documents the recommended branch protection rules

## Main Branch Protection

### Required Status Checks
- Continuous integration (CI) pipeline
- Code quality checks (flake8, black, mypy)
- Test coverage (minimum 80%)
- Security scanning (CodeQL)
- Performance benchmarks
- Documentation build

### Required Reviews
- At least 2 approving reviews
- Dismiss stale reviews
- Require review from code owners
- Allow self-approve for minor changes

### Enforcement Rules
- Restrict pushes to administrators
- Restrict deletions
- Require branches to be up to date
- Include administrators in restrictions

## Development Branch Protection

### Required Status Checks
- Basic CI pipeline
- Unit tests
- Code format checks

### Required Reviews
- At least 1 approving review

## Release Branch Protection

### Required Status Checks
- Full CI pipeline
- Security scanning
- Documentation build
- Package building
- Performance benchmarks

### Required Reviews
- At least 3 approving reviews
- Release manager approval required
"""
        
        (self.project_root / ".github" / "BRANCH_PROTECTION.md").write_text(protection_config)
        
    def create_workflow_dispatch_config(self) -> None:
        """Create manual workflow dispatch configuration"""
        workflow_dispatch_yml = """name: Manual Workflows

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
      verbose:
        description: 'Enable verbose logging'
        required: false
        default: false
        type: boolean
      test_level:
        description: 'Test level to run'
        required: false
        default: 'full'
        type: choice
        options:
          - quick
          - full
          - comprehensive

jobs:
  manual-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"
        
    - name: Run tests
      run: |
        if [ "${{ github.event.inputs.test_level }}" = "quick" ]; then
          python demo_quick_test.py
        elif [ "${{ github.event.inputs.test_level }}" = "full" ]; then
          python -m pytest tests/test_core.py -v
        else
          python -m pytest tests/ --cov=brain_ai
        fi
        echo "Tests completed for environment: ${{ github.event.inputs.environment }}"
"""
        
        (self.github_dir / "manual.yml").write_text(workflow_dispatch_yml)
        
    def create_labels(self) -> None:
        """Create label configuration for the repository"""
        labels_config = {
            "labels": [
                {
                    "name": "bug",
                    "description": "Something isn't working",
                    "color": "d73a4a"
                },
                {
                    "name": "enhancement",
                    "description": "New feature or request",
                    "color": "a2eeef"
                },
                {
                    "name": "documentation",
                    "description": "Improvements or additions to documentation",
                    "color": "0075ca"
                },
                {
                    "name": "good first issue",
                    "description": "Good for newcomers",
                    "color": "7057ff"
                },
                {
                    "name": "help wanted",
                    "description": "Extra attention is needed",
                    "color": "008672"
                },
                {
                    "name": "question",
                    "description": "Further information is requested",
                    "color": "d876e3"
                },
                {
                    "name": "wontfix",
                    "description": "This will not be worked on",
                    "color": "ffffff"
                },
                {
                    "name": "duplicate",
                    "description": "This issue or PR already exists",
                    "color": "cccccc"
                },
                {
                    "name": "performance",
                    "description": "Performance-related issues",
                    "color": "fbca04"
                },
                {
                    "name": "security",
                    "description": "Security vulnerabilities",
                    "color": "d73a4a"
                },
                {
                    "name": "memory",
                    "description": "Related to memory systems",
                    "color": "0e8a16"
                },
                {
                    "name": "continual learning",
                    "description": "Related to continual learning",
                    "color": "bfdadc"
                },
                {
                    "name": "attention",
                    "description": "Related to attention mechanisms",
                    "color": "c5def5"
                },
                {
                    "name": "hippocampus",
                    "description": "Related to hippocampus module",
                    "color": "f9c2e2"
                },
                {
                    "name": "neocortex",
                    "description": "Related to neocortex module",
                    "color": "c1f0d6"
                },
                {
                    "name": "needs-triage",
                    "description": "New issues that need initial assessment",
                    "color": "fbca04"
                },
                {
                    "name": "ready-for-review",
                    "description": "Ready for code review",
                    "color": "0e8a16"
                },
                {
                    "name": "in-review",
                    "description": "Currently under review",
                    "color": "fb8c00"
                },
                {
                    "name": "changes-requested",
                    "description": "Changes have been requested",
                    "color": "e11d21"
                },
                {
                    "name": "blocked",
                    "description": "Blocked by other issues or PRs",
                    "color": "d73a4a"
                },
                {
                    "name": "breaking changes",
                    "description": "Contains breaking changes",
                    "color": "d73a4a"
                },
                {
                    "name": "dependencies",
                    "description": "Updates to dependencies",
                    "color": "0366d6"
                },
                {
                    "name": "research",
                    "description": "Research and development",
                    "color": "6f42c1"
                },
                {
                    "name": "demo",
                    "description": "Demo and examples",
                    "color": "28a745"
                }
            ]
        }
        
        (self.project_root / ".github" / "labels.json").write_text(json.dumps(labels_config, indent=2))
        
    def create_milestones_config(self) -> None:
        """Create milestone configuration"""
        milestones_config = {
            "milestones": [
                {
                    "title": "v2.1.0 - Memory Consolidation",
                    "description": "Enhanced memory consolidation algorithms and improved performance",
                    "state": "open",
                    "due_on": "2025-03-01T00:00:00Z"
                },
                {
                    "title": "v2.2.0 - Performance Optimization",
                    "description": "Major performance improvements and GPU acceleration",
                    "state": "open",
                    "due_on": "2025-06-01T00:00:00Z"
                },
                {
                    "title": "v2.3.0 - Research Integration",
                    "description": "Integration of latest neuroscience research findings",
                    "state": "open",
                    "due_on": "2025-09-01T00:00:00Z"
                },
                {
                    "title": "v3.0.0 - Architecture Redesign",
                    "description": "Major architectural improvements and new features",
                    "state": "open",
                    "due_on": "2025-12-01T00:00:00Z"
                }
            ]
        }
        
        (self.project_root / ".github" / "milestones.json").write_text(json.dumps(milestones_config, indent=2))
        
    def create_contributors_md(self) -> None:
        """Create contributors list"""
        contributors_md = """# Contributors

We deeply appreciate all contributors who have helped make Brain-Inspired AI possible!

## Core Team 👑

### Maintainers
- **Jane Smith** - *Lead Researcher* - [@janesmith](https://github.com/janesmith)
- **John Doe** - *Core Developer* - [@johndoe](https://github.com/johndoe)
- **Alice Johnson** - *Scientific Advisor* - [@alicej](https://github.com/alicej)

### Contributors
<!-- This section is automatically updated by GitHub Actions -->

## Special Thanks 🙏

### Beta Testers
- Beta testing team and early adopters

### Documentation Contributors
- Documentation writers and translators

### Research Collaborators
- University partners and research collaborators

### Community Contributors
- All community members who provide feedback and support

## How to Become a Contributor

1. **Start Small**: Look for issues labeled "good first issue"
2. **Contribute Code**: Submit pull requests with new features or bug fixes
3. **Improve Documentation**: Help improve docs, examples, and tutorials
4. **Help Others**: Answer questions and provide support
5. **Share Knowledge**: Write blog posts, give talks, or create tutorials

## Recognition

Contributors are recognized in:
- This file
- Release notes
- Conference presentations
- Annual contributor report

## Join Us!

Want to contribute? Check out our [Contributing Guide](CONTRIBUTING.md) and [Good First Issues](https://github.com/your-repo/brain-inspired-ai/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

---

*This file is automatically updated by our GitHub Actions workflow every release.*

**Thank you for being part of the Brain-Inspired AI community! 🧠**
"""
        
        (self.project_root / "CONTRIBUTORS.md").write_text(contributors_md)
        
    def create_code_of_conduct(self) -> None:
        """Create Code of Conduct"""
        coc_md = """# Code of Conduct

## Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.

## Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event. Representation of a project may be further defined and clarified by project maintainers.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@brain-ai.org. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident. Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4, available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
"""
        
        (self.project_root / "CODE_OF_CONDUCT.md").write_text(coc_md)
        
    def setup_repository(self, comprehensive: bool = False) -> None:
        """Set up complete GitHub repository structure"""
        print("🚀 Setting up GitHub repository structure...")
        
        # Create main .github directory
        self.github_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.github_dir / "workflows").mkdir(exist_ok=True)
        (self.github_dir / "ISSUE_TEMPLATE").mkdir(exist_ok=True)
        
        print("📋 Creating issue templates...")
        self.create_issue_templates()
        
        print("🛡️ Creating security policy...")
        self.create_security_policy()
        
        print("💰 Creating funding configuration...")
        self.create_funding_yml()
        
        print("🔍 Creating security analysis config...")
        self.create_codeql_config()
        
        print("⚡ Creating stale issues config...")
        self.create_stale_issues_config()
        
        print("🤖 Creating auto-merge config...")
        self.create_auto_merge_config()
        
        print("🌿 Creating branch protection config...")
        self.create_branch_protection_config()
        
        print("🎮 Creating manual workflow config...")
        self.create_workflow_dispatch_config()
        
        print("🏷️ Creating label configuration...")
        self.create_labels()
        
        print("🎯 Creating milestone configuration...")
        self.create_milestones_config()
        
        if comprehensive:
            print("👥 Creating contributors list...")
            self.create_contributors_md()
            
            print("📜 Creating Code of Conduct...")
            self.create_code_of_conduct()
            
            print("🔄 Creating Dependabot config...")
            self.create_dependabot_config()
        
        print("✅ GitHub repository structure created successfully!")
        
        # Print summary
        print("\n📊 Created Files Summary:")
        print("├── .github/")
        print("│   ├── workflows/")
        print("│   │   ├── ci-cd.yml")
        print("│   │   ├── testing.yml")
        print("│   │   ├── release.yml")
        print("│   │   ├── docs.yml")
        print("│   │   ├── pages.yml")
        print("│   │   ├── stale.yml")
        print("│   │   ├── auto-merge.yml")
        print("│   │   ├── manual.yml")
        print("│   │   ├── codeql.yml")
        print("│   │   └── dependabot.yml")
        print("│   ├── ISSUE_TEMPLATE/")
        print("│   │   ├── bug_report.md")
        print("│   │   ├── feature_request.md")
        print("│   │   ├── performance_issue.md")
        print("│   │   ├── question.md")
        print("│   │   └── config.yml")
        print("│   ├── labels.json")
        print("│   ├── milestones.json")
        print("│   ├── BRANCH_PROTECTION.md")
        print("│   └── FUNDING.yml")
        print("├── SECURITY.md")
        print("├── CONTRIBUTING.md")
        print("├── CONTRIBUTORS.md")
        print("├── CODE_OF_CONDUCT.md")
        print("├── mkdocs.yml")
        print("└── pull_request_template.md")


def main():
    parser = argparse.ArgumentParser(description="GitHub Repository Setup Script")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Include additional files like CODE_OF_CONDUCT and contributors")
    parser.add_argument("--project-root", type=Path, 
                       help="Project root directory (default: current directory)")
    
    args = parser.parse_args()
    
    if args.project_root:
        project_root = args.project_root
    else:
        project_root = Path.cwd()
    
    print(f"🎯 Setting up GitHub repository at: {project_root}")
    
    setup = GitHubSetup(project_root)
    setup.setup_repository(comprehensive=args.comprehensive)
    
    print("\n🎉 GitHub repository setup completed!")
    print("\n📋 Next Steps:")
    print("1. Commit and push the changes")
    print("2. Configure branch protection rules in GitHub")
    print("3. Set up GitHub Pages (if needed)")
    print("4. Configure repository settings")
    print("5. Update repository description and topics")


if __name__ == "__main__":
    main()