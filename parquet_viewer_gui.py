import os
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------------
# PySimpleGUI 4.60.5 (LGPL, free) fallback to FreeSimpleGUI
# ----------------------------------------------------------------------
try:
    import PySimpleGUI as sg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import FreeSimpleGUI as sg  # type: ignore

sg.theme("DarkBlue14")

HELP = """
Parquet Viewer GUI (free)
-------------------------
• Works with **PySimpleGUI 4.60.5** – last LGPL release – or the drop‑in fork **FreeSimpleGUI**.
• Drag a .parquet file onto the path box *or* click “浏览”.
• Adjust the spin‑box to change how many head‑rows are shown.
• Window remains *on top* so you can toss files at it any time.
"""

def load_preview(path: str, n_head: int) -> Optional[pd.DataFrame]:
    """Read a parquet file and return the head slice or None on error."""
    if not os.path.isfile(path):
        sg.popup_error(f"文件不存在: {path}")
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            sg.popup("文件为空", title="信息")
            return None
        return df.head(n_head)
    except Exception as err:  # noqa: BLE001
        sg.popup_error(f"读取失败: {err}")
        return None


def build_window() -> sg.Window:
    """Initial window *without* the table; table gets injected after first load."""
    layout = [
        [sg.Text("显示前"),
         sg.Spin(values=(5, 10, 20, 50, 100, 200),
                 initial_value=10,
                 key="-NROWS-",
                 enable_events=True,
                 size=(5, 1)),
         sg.Text("行")],
        [sg.Input(key="-FILE-",
                  enable_events=True,  # fires when Browse writes or user presses ↵
                  tooltip="拖拽 Parquet 文件或点击右侧 ‘浏览’ 按钮",
                  readonly=True,
                  expand_x=True),
         sg.FileBrowse("浏览",
                       file_types=(("Parquet 文件", "*.parquet"),),
                       target="-FILE-")],
        # Placeholder column where the table will be inserted later
        [sg.Column([[]], key="-TABLE_COL-", expand_x=True, expand_y=True)],
        [sg.Button("退出", key="Exit")],
    ]
    return sg.Window(
        "Parquet 查看器 (Free)",
        layout,
        finalize=True,
        keep_on_top=True,
        resizable=True,
        return_keyboard_events=True,
    )


def inject_table(window: sg.Window, df: pd.DataFrame, n_rows: int) -> None:
    """Create the Table element the first time or reuse & update existing one."""
    values = df.head(n_rows).values.tolist()
    if "-TABLE-" in window.key_dict:
        window["-TABLE-"].update(values=values)
    else:
        table_elem = sg.Table(
            values=values,
            headings=list(df.columns),
            key="-TABLE-",
            num_rows=20,
            auto_size_columns=True,
            expand_x=True,
            expand_y=True,
            justification="left",
        )
        # Inject into the placeholder column
        window.extend_layout(window["-TABLE_COL-"], [[table_elem]])
        # Force window to readjust layout
        window.refresh()


def main() -> None:
    window = build_window()
    preview_df: Optional[pd.DataFrame] = None

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        # Spin‑box changed
        if event == "-NROWS-" and preview_df is not None:
            inject_table(window, preview_df, int(values["-NROWS-"]))

        # File picked or dragged
        if event == "-FILE-":
            file_path = values["-FILE-"]
            if file_path:
                n_head = int(values["-NROWS-"])
                preview_df = load_preview(file_path, n_head)
                if preview_df is not None:
                    inject_table(window, preview_df, n_head)

    window.close()


if __name__ == "__main__":
    main()
