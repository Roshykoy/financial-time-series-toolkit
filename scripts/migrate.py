#!/usr/bin/env python3
"""
Migration script for MarkSix Probabilistic Forecasting System reorganization.
Safely migrates from the old structure to the new modular architecture.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
import argparse


class MigrationManager:
    """Manages the migration process from old to new structure."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.backup_dir = project_root / "backup_old_structure"
        
    def run_migration(self) -> None:
        """Execute the complete migration process."""
        print("🚀 Starting MarkSix Project Structure Migration")
        print("=" * 50)
        
        if self.dry_run:
            print("⚠️  DRY RUN MODE - No files will be modified")
        
        try:
            # Step 1: Create backup
            self._create_backup()
            
            # Step 2: Create new directory structure
            self._create_directory_structure()
            
            # Step 3: Create compatibility layer
            self._create_compatibility_layer()
            
            # Step 4: Update imports gradually
            self._update_main_imports()
            
            print("\n✅ Migration completed successfully!")
            print("📝 Next steps:")
            print("1. Test the system with: python main.py")
            print("2. Update imports in existing modules gradually")
            print("3. Move model implementations to core/models/")
            print("4. Update training/inference pipelines to use services")
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            print("🔄 Restoring from backup...")
            self._restore_backup()
            raise
    
    def _create_backup(self) -> None:
        """Create backup of existing structure."""
        print("\n📦 Creating backup of existing structure...")
        
        if self.backup_dir.exists():
            if not self.dry_run:
                shutil.rmtree(self.backup_dir)
        
        if not self.dry_run:
            self.backup_dir.mkdir(exist_ok=True)
            
            # Backup key files and directories
            backup_items = [
                'src/config.py',
                'src/config_manager.py', 
                'main.py'
            ]
            
            for item in backup_items:
                src_path = self.project_root / item
                if src_path.exists():
                    dest_path = self.backup_dir / item
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    if src_path.is_file():
                        shutil.copy2(src_path, dest_path)
                    else:
                        shutil.copytree(src_path, dest_path)
        
        print("✅ Backup created")
    
    def _create_directory_structure(self) -> None:
        """Create the new directory structure."""
        print("\n📁 Creating new directory structure...")
        
        directories = [
            'src/core/models',
            'src/core/data',
            'src/core/training',
            'src/core/inference',
            'src/infrastructure/config',
            'src/infrastructure/logging',
            'src/infrastructure/storage',
            'src/infrastructure/monitoring',
            'src/application/services',
            'src/application/cli/commands',
            'src/utils',
            'config/environments',
            'config/model_presets',
            'tests/unit',
            'tests/integration',
            'tests/fixtures',
            'docs/api',
            'scripts',
            'requirements'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            if not self.dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
                # Create __init__.py files for Python packages
                if directory.startswith('src/'):
                    init_file = dir_path / '__init__.py'
                    if not init_file.exists():
                        init_file.write_text('"""Package initialization."""\n')
        
        print(f"✅ Created {len(directories)} directories")
    
    def _create_compatibility_layer(self) -> None:
        """Create compatibility layer for gradual migration."""
        print("\n🔄 Creating compatibility layer...")
        
        if not self.dry_run:
            # Create legacy config bridge
            compat_config = self.project_root / 'src' / 'config_legacy.py'
            compat_config.write_text('''"""
Legacy configuration bridge for backward compatibility.
This file provides the old CONFIG interface while using the new system.
"""
import warnings
from src.infrastructure.config import get_flat_config

# Issue deprecation warning
warnings.warn(
    "Direct import of CONFIG is deprecated. Use get_config() from src.infrastructure.config instead.",
    DeprecationWarning,
    stacklevel=2
)

# Provide backward-compatible CONFIG object
CONFIG = get_flat_config()
''')
            
            # Update old config.py to use new system
            old_config = self.project_root / 'src' / 'config.py'
            if old_config.exists():
                old_config.rename(self.project_root / 'src' / 'config_original.py')
            
            # Create new config.py that imports from new location
            old_config.write_text('''"""
Configuration module - migrated to new structure.
Import from src.infrastructure.config for new code.
"""
from src.config_legacy import CONFIG

# Re-export for compatibility
__all__ = ['CONFIG']
''')
        
        print("✅ Compatibility layer created")
    
    def _update_main_imports(self) -> None:
        """Update main.py to use new configuration system."""
        print("\n📝 Updating main.py imports...")
        
        main_file = self.project_root / 'main.py'
        if not main_file.exists():
            print("⚠️  main.py not found, skipping")
            return
        
        if not self.dry_run:
            # Read current content
            content = main_file.read_text()
            
            # Add new import at the top (after existing imports)
            new_imports = '''
# Enhanced configuration and logging
from src.infrastructure.config import get_config_manager, configure_logging
from src.infrastructure.logging import get_logger

# Initialize enhanced systems
configure_logging(log_level="INFO", log_file="marksix.log")
logger = get_logger(__name__)
'''
            
            # Find where to insert (after last import)
            lines = content.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    insert_pos = i + 1
            
            # Insert new imports
            lines.insert(insert_pos, new_imports)
            
            # Write back
            main_file.write_text('\n'.join(lines))
        
        print("✅ main.py updated")
    
    def _restore_backup(self) -> None:
        """Restore from backup in case of failure."""
        if not self.backup_dir.exists():
            return
        
        print("🔄 Restoring from backup...")
        
        # Restore backed up files
        for item in self.backup_dir.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(self.backup_dir)
                target_path = self.project_root / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_path)
        
        print("✅ Backup restored")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate MarkSix project structure")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    
    args = parser.parse_args()
    
    # Validate project root
    if not (args.project_root / 'src').exists():
        print(f"❌ Invalid project root: {args.project_root}")
        print("Please run from the MarkSix project root directory")
        sys.exit(1)
    
    # Run migration
    migrator = MigrationManager(args.project_root, dry_run=args.dry_run)
    try:
        migrator.run_migration()
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()