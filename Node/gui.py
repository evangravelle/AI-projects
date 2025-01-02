import FreeSimpleGUI as sg
import game_state

SQUARE_LEN = 128
WINDOW_SIDE_BUFFER = 128
GRAPH_LEN = 5 * SQUARE_LEN + 2 * WINDOW_SIDE_BUFFER
NODE_RADIUS = int(0.17 * SQUARE_LEN)
ROAD_LEN = int(0.58 * SQUARE_LEN)
ROAD_WIDTH = int(0.1 * SQUARE_LEN)

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


def coord_to_pix(coordinates):
    return (
        WINDOW_SIDE_BUFFER + SQUARE_LEN * coordinates[0],
        WINDOW_SIDE_BUFFER + SQUARE_LEN * coordinates[1],
    )


def create_window(button_str="Create Board"):
    layout = [
        [
            sg.Graph(
                canvas_size=(GRAPH_LEN, GRAPH_LEN),
                graph_bottom_left=(0, 0),
                graph_top_right=(GRAPH_LEN, GRAPH_LEN),
                key="-GRAPH-",
            )
        ],
        [sg.Button(button_str, key="-BUTTON-")],
    ]

    window = sg.Window(title="Node", layout=layout)
    return window


def draw_board(window, rng_seed=0):
    state = game_state.GameState(rng_seed=rng_seed)

    asset_dir = "C:\\Users\\evang\\Desktop\\git\\AI-projects\\Node\\assets\\"
    graph = window["-GRAPH-"]
    for sq_key, sq_val in state.board.items():
        pix = coord_to_pix(sq_key)
        x = pix[0]
        y = SQUARE_LEN + pix[1]
        filepath = asset_dir + SQUARE_FILENAME[sq_val]

        # location of top left corner of image
        graph.draw_image(filename=filepath, location=(x, y))

        # sg.Graph.draw_circle()
        # draw_node(graph=graph, coord=(2, 2), color="orange")

        # sg.Graph.draw_rectangle()
        # draw_road(graph=graph, coord=(2, 1.5), color="orange")


def draw_node(graph, coord, color):
    pix = coord_to_pix(coord)
    line_width = 4
    node_id = graph.draw_circle(pix, radius=NODE_RADIUS, fill_color=color, line_width=line_width)
    return node_id


def draw_road(graph, coord, color):
    if coord is None:
        return

    if coord[0] - int(coord[0]) == 0.5:
        # Road should be horizontal
        dx = ROAD_LEN
        dy = ROAD_WIDTH
    elif coord[1] - int(coord[1]) == 0.5:
        # Road should be vertical
        dx = ROAD_WIDTH
        dy = ROAD_LEN
    else:
        raise Exception("One of the road coordinates needs to end in 0.5")

    pix = coord_to_pix(coord)
    top_left = (pix[0] - dx / 2.0, pix[1] + dy / 2.0)
    bottom_right = (pix[0] + dx / 2.0, pix[1] - dy / 2.0)
    line_width = 4
    road_id = graph.draw_rectangle(
        top_left=top_left,
        bottom_right=bottom_right,
        fill_color=color,
        line_width=line_width,
    )
    return road_id


def event_loop(window):
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == "-BUTTON-":
            import time

            ms = int(time.time() * 1000.0)
            draw_board(window, rng_seed=ms)

    window.close()


def main():
    window = create_window()
    event_loop(window)


if __name__ == "__main__":
    # sg.main_sdk_help()
    main()
