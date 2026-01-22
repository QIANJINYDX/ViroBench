"""
Project-local datasets package.

Note: This exists to avoid import-resolution issues with the external `datasets`
package (HuggingFace). When `PROJECT_ROOT` is placed at the front of `sys.path`,
`from datasets...` will resolve to this package.
"""

