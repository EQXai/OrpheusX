from orpheusx.ui.gradio_ui import build_ui


def main():
    demo = build_ui()
    demo.queue()
    demo.launch(server_port=18188)


if __name__ == "__main__":
    main() 