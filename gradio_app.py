#!/usr/bin/env python
"""Backwards-compatibility wrapper.

The original monolithic *gradio_app.py* has been refactored into a
package-based structure under :pymod:`orpheusx`.  This stub simply
delegates execution to :pyfunc:`orpheusx.app.main` so that existing
workflows—or documentation that instructs running ``python gradio_app.py``—
continue to operate unchanged.
"""
from orpheusx.app import main

if __name__ == "__main__":
    main() 