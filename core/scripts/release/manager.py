#!/usr/bin/env python3
"""
GitHub Release Management Script
自动化的版本发布和标签管理工具
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class ReleaseManager:
    """GitHub Release Manager for Brain-Inspired AI"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.setup_py = project_root / "pyproject.toml"
        self.version_file = project_root / "brain_ai" / "__init__.py"
        
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml or __init__.py"""
        try:
            with open(self.setup_py, "rb") as f:
                data = tomllib.load(f)
            return data["project"]["version"]
        except (FileNotFoundError, KeyError):
            # Fallback to __init__.py
            with open(self.version_file) as f:
                content = f.read()
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
                raise ValueError("Could not find version in pyproject.toml or __init__.py")
    
    def validate_version(self, version: str) -> bool:
        """Validate semantic versioning format"""
        pattern = r'^\d+\.\d+\.\d+(?:[a-zA-Z]+[0-9]*)?$'
        return bool(re.match(pattern, version))
    
    def get_version_parts(self, version: str) -> Dict[str, str]:
        """Extract version parts (major, minor, patch, prerelease)"""
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:([a-zA-Z]+[0-9]*))?$', version)
        if not match:
            raise ValueError(f"Invalid version format: {version}")
        
        parts = {
            "major": match.group(1),
            "minor": match.group(2),
            "patch": match.group(3),
        }
        if match.group(4):
            parts["prerelease"] = match.group(4)
        return parts
    
    def increment_version(self, version: str, part: str = "patch") -> str:
        """Increment version by specified part"""
        parts = self.get_version_parts(version)
        
        if part == "major":
            parts["major"] = str(int(parts["major"]) + 1)
            parts["minor"] = "0"
            parts["patch"] = "0"
        elif part == "minor":
            parts["minor"] = str(int(parts["minor"]) + 1)
            parts["patch"] = "0"
        elif part == "patch":
            parts["patch"] = str(int(parts["patch"]) + 1)
        else:
            raise ValueError(f"Invalid version part: {part}")
        
        # Remove prerelease if incrementing
        if "prerelease" in parts:
            del parts["prerelease"]
        
        return f'{parts["major"]}.{parts["minor"]}.{parts["patch"]}'
    
    def update_version_files(self, new_version: str) -> None:
        """Update version in all relevant files"""
        # Update pyproject.toml
        if self.setup_py.exists():
            with open(self.setup_py, "rb") as f:
                data = tomllib.load(f)
            data["project"]["version"] = new_version
            
            with open(self.setup_py, "wb") as f:
                import tomli_w
                tomli_w.dump(data, f)
        
        # Update __init__.py
        if self.version_file.exists():
            with open(self.version_file) as f:
                content = f.read()
            
            content = re.sub(
                r'__version__\s*=\s*["\'][^"\']+["\']',
                f'__version__ = "{new_version}"',
                content
            )
            
            with open(self.version_file, "w") as f:
                f.write(content)
    
    def generate_changelog(self, from_version: Optional[str] = None) -> str:
        """Generate changelog from git history"""
        cmd = ["git", "log", "--pretty=format:%H|%s|%an|%ad", "--date=short"]
        
        if from_version:
            try:
                # Try to get the tag
                subprocess.run(["git", "rev-parse", from_version], check=True, capture_output=True)
                cmd.append(from_version + "..HEAD")
            except subprocess.CalledProcessError:
                print(f"Warning: Could not find tag {from_version}, using all history")
        else:
            # Get last tag
            try:
                last_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], 
                                                 text=True).strip()
                cmd.append(last_tag + "..HEAD")
            except subprocess.CalledProcessError:
                print("Warning: No previous tags found, using all history")
        
        try:
            output = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating changelog: {e}")
            return "No changes to report."
        
        lines = output.strip().split('\n') if output.strip() else []
        
        # Categorize changes
        changes = {
            "feat": [],  # New features
            "fix": [],   # Bug fixes
            "docs": [],  # Documentation
            "refactor": [],  # Code refactoring
            "test": [],  # Tests
            "chore": []  # Maintenance
        }
        
        for line in lines:
            if '|' in line:
                commit_hash, message, author, date = line.split('|')
                
                # Categorize based on commit message
                if message.startswith('feat:'):
                    changes["feat"].append(f"- {message[5:]} ({commit_hash[:7]})")
                elif message.startswith('fix:'):
                    changes["fix"].append(f"- {message[4:]} ({commit_hash[:7]})")
                elif any(prefix in message for prefix in ['docs:', 'doc:']):
                    changes["docs"].append(f"- {message[message.find(':')+1:]} ({commit_hash[:7]})")
                elif any(prefix in message for prefix in ['refactor:', 'perf:']):
                    changes["refactor"].append(f"- {message[message.find(':')+1:]} ({commit_hash[:7]})")
                elif any(prefix in message for prefix in ['test:', 'test']):
                    changes["test"].append(f"- {message[message.find(':')+1:]} ({commit_hash[:7]})")
                else:
                    changes["chore"].append(f"- {message} ({commit_hash[:7]})")
        
        # Generate markdown
        changelog = f"# Changelog\n\nGenerated on {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        if any(changes.values()):
            for category, items in changes.items():
                if items:
                    title = {
                        "feat": "✨ New Features",
                        "fix": "🐛 Bug Fixes",
                        "docs": "📚 Documentation",
                        "refactor": "🔧 Refactoring",
                        "test": "🧪 Tests",
                        "chore": "🔧 Maintenance"
                    }[category]
                    changelog += f"\n## {title}\n\n"
                    changelog += "\n".join(items) + "\n"
        else:
            changelog += "\nNo significant changes since the last release.\n"
        
        return changelog
    
    def create_tag(self, version: str) -> None:
        """Create and push git tag"""
        tag_name = f"v{version}"
        
        # Check if tag already exists
        try:
            subprocess.run(["git", "rev-parse", tag_name], check=True, capture_output=True)
            print(f"Tag {tag_name} already exists")
            return
        except subprocess.CalledProcessError:
            pass
        
        # Create tag
        message = f"Release v{version}"
        subprocess.run(["git", "tag", "-a", tag_name, "-m", message], check=True)
        print(f"Created tag {tag_name}")
        
        # Push tag
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        print(f"Pushed tag {tag_name} to origin")
    
    def run_tests(self) -> bool:
        """Run the test suite"""
        try:
            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("Tests failed:")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print("✅ All tests passed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error running tests: {e}")
            return False
    
    def build_package(self) -> bool:
        """Build source and wheel distributions"""
        try:
            # Clean previous builds
            for dir_name in ["dist", "build", "*.egg-info"]:
                subprocess.run(["rm", "-rf", dir_name], cwd=self.project_root)
            
            # Build package
            subprocess.run(
                ["python", "-m", "build"],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            # Verify package
            subprocess.run(
                ["python", "-m", "twine", "check", "dist/*"],
                cwd=self.project_root,
                check=True
            )
            
            print("✅ Package built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error building package: {e}")
            return False
    
    def release(self, version: str, dry_run: bool = False) -> None:
        """Execute the full release process"""
        print(f"🚀 Starting release process for version {version}")
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        
        # Validate version format
        if not self.validate_version(version):
            raise ValueError(f"Invalid version format: {version}")
        
        current_version = self.get_current_version()
        print(f"Current version: {current_version}")
        
        # Check if this is a new version
        if version == current_version:
            if not dry_run:
                print("Version already updated. Continuing with release...")
        else:
            print(f"📝 Updating version from {current_version} to {version}")
            if not dry_run:
                self.update_version_files(version)
        
        # Run tests
        print("🧪 Running tests...")
        if not self.run_tests():
            raise RuntimeError("Tests failed. Aborting release.")
        
        # Generate changelog
        print("📝 Generating changelog...")
        changelog = self.generate_changelog(current_version)
        changelog_file = self.project_root / "CHANGELOG_RELEASE.md"
        
        if not dry_run:
            with open(changelog_file, "w") as f:
                f.write(changelog)
            print(f"Changelog saved to {changelog_file}")
        
        # Build package
        print("📦 Building package...")
        if not self.build_package():
            raise RuntimeError("Package build failed. Aborting release.")
        
        # Create tag
        if not dry_run:
            print(f"🏷️ Creating tag v{version}...")
            self.create_tag(version)
        
        print("🎉 Release process completed successfully!")
        
        if not dry_run:
            print("\nNext steps:")
            print("1. Review the changelog")
            print("2. Push the changes: git push origin main && git push origin --tags")
            print("3. Create a GitHub release manually or run the release workflow")
            print("4. Publish to PyPI: python -m twine upload dist/*")


def main():
    parser = argparse.ArgumentParser(description="Brain-Inspired AI Release Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Version commands
    version_parser = subparsers.add_parser("version", help="Show or manage versions")
    version_parser.add_argument("--show", action="store_true", help="Show current version")
    version_parser.add_argument("--increment", choices=["major", "minor", "patch"], 
                               help="Increment version by specified part")
    version_parser.add_argument("--set", help="Set specific version")
    
    # Release command
    release_parser = subparsers.add_parser("release", help="Execute release process")
    release_parser.add_argument("--version", help="Release version (e.g., 1.2.3)")
    release_parser.add_argument("--auto", action="store_true", 
                               help="Auto-increment patch version")
    release_parser.add_argument("--dry-run", action="store_true", 
                               help="Preview release without making changes")
    
    # Changelog command
    changelog_parser = subparsers.add_parser("changelog", help="Generate changelog")
    changelog_parser.add_argument("--from", dest="from_version", 
                                 help="Generate changelog from this version")
    changelog_parser.add_argument("--output", help="Output file path")
    
    # Tag command
    tag_parser = subparsers.add_parser("tag", help="Manage tags")
    tag_parser.add_argument("--create", help="Create tag for version")
    tag_parser.add_argument("--list", action="store_true", help="List all tags")
    tag_parser.add_argument("--push", action="store_true", help="Push tags to remote")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    release_manager = ReleaseManager(project_root)
    
    if args.command == "version":
        if args.show:
            print(release_manager.get_current_version())
        elif args.increment:
            current = release_manager.get_current_version()
            new_version = release_manager.increment_version(current, args.increment)
            print(f"{current} -> {new_version}")
        elif args.set:
            if release_manager.validate_version(args.set):
                release_manager.update_version_files(args.set)
                print(f"Version updated to {args.set}")
            else:
                print(f"Invalid version format: {args.set}")
        else:
            parser.print_help()
    
    elif args.command == "release":
        if args.auto:
            current = release_manager.get_current_version()
            version = release_manager.increment_version(current, "patch")
        elif args.version:
            version = args.version
        else:
            parser.error("Either --version or --auto must be specified")
        
        try:
            release_manager.release(version, dry_run=args.dry_run)
        except Exception as e:
            print(f"❌ Release failed: {e}")
            sys.exit(1)
    
    elif args.command == "changelog":
        changelog = release_manager.generate_changelog(args.from_version)
        if args.output:
            with open(args.output, "w") as f:
                f.write(changelog)
            print(f"Changelog written to {args.output}")
        else:
            print(changelog)
    
    elif args.command == "tag":
        if args.list:
            try:
                output = subprocess.check_output(["git", "tag", "-l"], text=True)
                print("Available tags:")
                for line in output.strip().split('\n'):
                    if line:
                        print(f"  {line}")
            except subprocess.CalledProcessError:
                print("No tags found")
        elif args.create:
            if release_manager.validate_version(args.create):
                if not args.create.startswith('v'):
                    version = f"v{args.create}"
                else:
                    version = args.create
                release_manager.create_tag(version)
            else:
                print(f"Invalid version format: {args.create}")
        elif args.push:
            try:
                subprocess.run(["git", "push", "origin", "--tags"], check=True)
                print("All tags pushed to origin")
            except subprocess.CalledProcessError:
                print("Failed to push tags")
        else:
            parser.print_help()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()