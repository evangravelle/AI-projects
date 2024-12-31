import FreeSimpleGUI as sg
import game_state

# sg.main_sdk_help()

SQUARE_LEN = 128
WINDOW_SIDE_BUFFER = 128
GRAPH_LEN = 5 * SQUARE_LEN + 2 * WINDOW_SIDE_BUFFER

SQUARE_FILENAME = {
    "blank": "grey.png",
    "Y1": "yellow1.png",
    "Y2": "yellow2.png",
    "Y3": "yellow3.png",
    "G1": "green1.png",
    "G2": "green2.png",
    "G3": "green3.png",
    "R1": "red1.png",
    "R2": "red2.png",
    "R3": "red3.png",
    "B1": "blue1.png",
    "B2": "blue2.png",
    "B3": "blue3.png",
}


def create_window():
    layout = [
        [
            sg.Graph(
                canvas_size=(GRAPH_LEN, GRAPH_LEN),
                graph_bottom_left=(0, 0),
                graph_top_right=(GRAPH_LEN, GRAPH_LEN),
                key="-GRAPH-",
            )
        ],
        [sg.Button("Create Board")],
    ]

    window = sg.Window(title="Node", layout=layout)
    return window


def create_board(window):
    state = game_state.GameState(rng_seed=1)
    # print("BOARD KEYS")
    # print(state.board.keys())
    # print("BOARD VALUES")
    # print(state.board.values())

    asset_dir = "C:\\Users\\evang\\Desktop\\node assets\\"
    graph = window["-GRAPH-"]
    for sq_key, sq_val in state.board.items():
        x = WINDOW_SIDE_BUFFER + sq_key[0] * SQUARE_LEN
        y = WINDOW_SIDE_BUFFER + sq_key[1] * SQUARE_LEN
        print("X = " + str(x))
        print("Y = " + str(y))
        filepath = asset_dir + SQUARE_FILENAME[sq_val]
        graph.draw_image(filename=filepath, location=(x, y))


def event_loop(window):
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == "Create Board":
            create_board(window)

    window.close()
