import FreeSimpleGUI as sg
import game_state

SQUARE_LEN = 128  # pixels
WINDOW_SIDE_BUFFER = 128  # pixels
GRAPH_LEN = 5 * SQUARE_LEN + 2 * WINDOW_SIDE_BUFFER  # pixels
NODE_RADIUS = int(0.17 * SQUARE_LEN)  # pixels
ROAD_LEN = int(0.58 * SQUARE_LEN)  # pixels
ROAD_WIDTH = int(0.1 * SQUARE_LEN)  # pixels

SQUARE_FILENAME = {
    "_": "grey.png",
    "y1": "yellow1.png",
    "y2": "yellow2.png",
    "y3": "yellow3.png",
    "g1": "green1.png",
    "g2": "green2.png",
    "g3": "green3.png",
    "r1": "red1.png",
    "r2": "red2.png",
    "r3": "red3.png",
    "b1": "blue1.png",
    "b2": "blue2.png",
    "b3": "blue3.png",
}

COLOR = ["orange", "purple"]


def coord_to_pix(coordinates):
    return (
        WINDOW_SIDE_BUFFER + SQUARE_LEN * coordinates[0],
        WINDOW_SIDE_BUFFER + SQUARE_LEN * coordinates[1],
    )


def pix_to_coord(pix):
    return (
        (pix[0] - WINDOW_SIDE_BUFFER) / SQUARE_LEN,
        (pix[1] - WINDOW_SIDE_BUFFER) / SQUARE_LEN,
    )


def create_window(button_str="Create Board"):
    layout = [
        [
            sg.Graph(
                canvas_size=(GRAPH_LEN, GRAPH_LEN),
                graph_bottom_left=(0, 0),
                graph_top_right=(GRAPH_LEN, GRAPH_LEN),
                enable_events=True,
                key="-GRAPH-",
            )
        ],
        [
            sg.Button(button_text=button_str, key="-BUTTON-"),
            sg.Push(),
            sg.Text(text=f"P1: Y={0}, G={0}, R={0}, B={0}", key="-P1-RESOURCES-"),
            sg.Push(),
            sg.Text(text=f"P2: Y={0}, G={0}, R={0}, B={0}", key="-P2-RESOURCES-"),
        ],
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
        y = pix[1]
        filepath = asset_dir + SQUARE_FILENAME[sq_val]

        # location of top left corner of image
        graph.draw_image(filename=filepath, location=(x, y))

    return state


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


def draw_resources(window, state):
    p1_text = (
        f"P1: Y={state.resources[0]["y"]}, G={state.resources[0]["g"]}, "
        f"R={state.resources[0]["r"]}, B={state.resources[0]["b"]}"
    )
    window["-P1-RESOURCES-"].update(p1_text)
    p2_text = (
        f"P2: Y={state.resources[1]["y"]}, G={state.resources[1]["g"]}, "
        f"R={state.resources[1]["r"]}, B={state.resources[1]["b"]}"
    )
    window["-P2-RESOURCES-"].update(p2_text)


# def is_inside_circle(pix, center, radius):
#     return (pix[0] - center[0]) ** 2 + (pix[1] - center[1]) ** 2 <= radius**2


def handle_graph_event(graph, pix, player):
    # Round to the nearest half coordinate
    coord_float = pix_to_coord(pix)
    coord = (round(coord_float[0] * 2) / 2.0, round(coord_float[1] * 2) / 2.0)
    print(coord)
    if coord in game_state.PLAYABLE_ROADS_DICT.keys():
        draw_road(graph, coord, color=COLOR[player])
        return coord
    elif 0 <= coord[0] <= 5 and 0 <= coord[1] <= 5 and coord not in game_state.UNPLAYABLE_NODES:
        draw_node(graph, coord, color=COLOR[player])
        return coord
    else:
        print(f"Unexpected graph event, pix={pix}, ignoring.")
        return None


def event_loop(window):
    state = None
    new_pieces = set()
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "-BUTTON-":
            if window["-BUTTON-"].ButtonText == "Create Board":
                import time

                ms = int(time.time() * 1000.0)
                state = draw_board(window, rng_seed=ms)
                window["-BUTTON-"].update(text="Finish Move")

            elif window["-BUTTON-"].ButtonText == "Finish Move":
                state.move(new_pieces)
                draw_resources(window, state)
                new_pieces.clear()

        elif event == "-GRAPH-":
            new_piece = handle_graph_event(window["-GRAPH-"], values["-GRAPH-"], state.player)
            if new_piece is not None:
                new_pieces.add(new_piece)

    window.close()


def main():
    window = create_window()
    event_loop(window)


if __name__ == "__main__":
    # sg.main_sdk_help()
    main()
