import numpy as np
import sys
from Tkinter import *
#import time
cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_f


@cython.boundscheck(False)
cdef int play_game(unsigned int a, unsigned int b, unsigned int x, unsigned int y,
    unsigned int p1, unsigned int p2, np.ndarray[DTYPE_f, ndim=2] scores_lattice,
    float d):
    """ Play a game between a and b,
    and update their scores.
    """

    #cdef unsigned int p1 = player_lattice[<unsigned int>a, <unsigned int>b]
    #cdef unsigned int p2 = player_lattice[<unsigned int>x, <unsigned int>y]

    if p1 == p2:
        scores_lattice[a, b] = scores_lattice[a, b] + p1
        scores_lattice[x, y] = scores_lattice[x, y] + p1
        return 0
    # If p2 defects, they get d
    if p1 > p2:
        scores_lattice[x, y] = scores_lattice[x, y] + d
        return 0
    if p1 < p2:
        scores_lattice[a, b] = scores_lattice[a, b] + d
        #return
    return 0
    #print a, b, p1,p,scores_lattice[a, b], "playing", p2, q, scores_lattice[x, y]


def neighbours(int x, int y):
    """ Get all the surrounding squares of square
    (x, y).
    """
    neighbours = []
    if toroid == 1:
        neighbours.append([(x + 1) % size, y])
        neighbours.append([(x + 1) % size, (y + 1) % size])
        neighbours.append([(x + 1) % size, (y - 1) % size])
        neighbours.append([x, (y + 1) % size])
        neighbours.append([(x - 1) % size, y])
        neighbours.append([(x - 1) % size, (y - 1) % size])
        neighbours.append([(x - 1) % size, (y + 1) % size])
        neighbours.append([x, (y - 1) % size])
        return neighbours

    if x < size - 1:
        neighbours.append([x + 1, y])
        if y < size - 1:
            neighbours.append([x + 1, y + 1])
        if y > 0:
            neighbours.append([x + 1, y - 1])
    if y < size - 1:
        neighbours.append([x, y + 1])
    if x > 0:
        neighbours.append([x - 1, y])
        if y > 0:
            neighbours.append([x - 1, y - 1])
        if y < size - 1:
            neighbours.append([x - 1, y + 1])
    if y > 0:
        neighbours.append([x, y - 1])
    return neighbours


@cython.boundscheck(False)
cdef float magnitude(np.ndarray[DTYPE_t, ndim = 1] vector):
    """ Return the magnitude of a vector.
    """

    return np.sqrt(vector.dot(vector))


def small_worldify(neighbours_lattice, float beta):
    """ Transform a regular lattice into a small world
    using the Watts-Strogatz method.
    """
    # Break some links
    # Rewrite this to be more consistent with the Watts-Strogatz method.
    cdef unsigned int new_links = 0
    cdef unsigned int x, y
    cdef float mag
    for x in range(size):
        for y in range(size):
            neighbours = []
            for neighbour in neighbours_lattice[x][y]:
                # Check the distance - only consider breaking links for direct neighbours
                mag = magnitude(np.array([x, y]) - np.array(neighbour))
                if np.random.rand() > beta or mag > 1.5:
                    neighbours.append(neighbour)
                else:
                    # print neighbours_lattice[neighbour[0]][neighbour[1]], x, y
                    # Remove from neighbour
                    #print neighbours_lattice[neighbour[0]][neighbour[1]], x, y
                    neighbours_lattice[neighbour[0]][neighbour[1]].remove([x, y])
                    # Add a link to a random neighbour (add a check here that this isn't a dupe)
                    neighbour = [np.random.randint(0, size), np.random.randint(0, size)]
                    neighbours.append(neighbour)
                    neighbours_lattice[neighbour[0]][neighbour[1]].append([x, y])
                    new_links = new_links + 1
            neighbours_lattice[x][y] = neighbours
    print "# Small-worldness factor %f" % beta
    print "# Broke", new_links, "links."

    return None

@cython.boundscheck(False)
cdef int find_winner(unsigned int x, unsigned int y, np.ndarray[DTYPE_t, ndim=2] player_lattice,
    np.ndarray[DTYPE_f, ndim=2] scores_lattice,
    np.ndarray neighbours_lattice):
    """ Find who has the highest score about a square
    and put them in that square.
    """
    cdef unsigned int a, b, p, q
    cdef float winner, challenger
    p = x
    q = y
    winner = scores_lattice[p, q]
    for [a, b] in neighbours_lattice[x, y]:
        challenger = scores_lattice[a, b]
        if challenger > winner:
            p = a
            q = b
            winner = scores_lattice[p, q]
    return player_lattice[p, q]


cdef int num_coop(np.ndarray[DTYPE_t, ndim=2] state):
    """ Get the number of cooperators
    in a game state.
    """
    return np.sum(state, dtype=int)


@cython.boundscheck(False)
cdef void play_round(np.ndarray[DTYPE_t, ndim=2] player_lattice, np.ndarray[DTYPE_f, ndim=2] scores_lattice,
    np.ndarray neighbours_lattice, float b, np.ndarray[DTYPE_t, ndim=2] last_player_lattice,
    np.ndarray[DTYPE_t, ndim=3] games):
    """ Play a round - everybody plays
    one game with each of their neighbours.
    """
    cdef unsigned int x, y, p1_x, p1_y, p2_x, p2_y, p1, p2
    #t_start = time.time()
    for x in range(num_games):
                #t_start_g = time.time()
                #print game
        p1_x = games[x, 0, 0]
        p1_y = games[x, 0, 1]
        p2_y = games[x, 1, 1]
        p2_x = games[x, 1, 0]
        p1 = player_lattice[p1_x, p1_y]
        p2 = player_lattice[p2_x, p2_y]
        play_game(p1_x, p1_y, p2_x, p2_y, p1, p2, scores_lattice, b)
                #print "Game %f seconds" % (time.time() - t_start_g)
    #print "Games %f seconds" % (time.time() - t_start)
    #t_start = time.time()
    for x in range(0, size):
        for y in range(0, size):
            #t_start = time.time()
            last_player_lattice[x, y] = player_lattice[x, y]
            #print "Winner %f seconds" % (time.time() - t_start)
    #print "Lattice update %f seconds" % (time.time() - t_start)
    #t_start = time.time()
    for x in range(0, size):
        for y in range(0, size):
            #t_start = time.time()
            player_lattice[x, y] = find_winner(x, y, last_player_lattice, scores_lattice, neighbours_lattice)
            #print "Winner %f seconds" % (time.time() - t_start)
    #print "Winners %f seconds" % (time.time() - t_start)


def init_gui(rectangles):
    """ Set up the gui.
    """
    master = Tk()
    w = Canvas(master, width=size * scale + scale, height=size * scale + scale)
    rectangles += np.array([[w.create_rectangle(scale * x, scale * y,
                    scale * x + scale, scale * y + scale,
                    fill="#000000", outline="#000000", width=0)
                 for x in range(size)] for y in range(size)], dtype=np.int)
    w.pack()
    return w


def update_gui(w, np.ndarray[DTYPE_t, ndim = 2] rectangles, np.ndarray[DTYPE_t, ndim = 2] player_lattice,
    np.ndarray[DTYPE_t, ndim = 2] last_player_lattice):
    """ Redraw the screen.
    """

    cdef str fill

    for x in range(size):
        for y in range(size):
                    # Cooperator
            if player_lattice[x, y] == 1 and last_player_lattice[x, y] == 1:
                fill = "blue"
                    # Defector
            elif player_lattice[x, y] == 0 and last_player_lattice[x, y] == 0:
                fill = "red"
            elif player_lattice[x, y] == 0 and last_player_lattice[x, y] == 1:
                fill = "yellow"
            else:
                fill = "green"

            w.itemconfig(rectangles[x, y], fill=fill)

        #print "Rects: %f seconds" % (time.time() - t_start)
    w.update()


def make_games(np.ndarray neighbours_lattice):
    """ Construct a list and set of the games that must be played in
    each round.
    """
    game_list = []
    game_set = []

    for x in range(size):
            for y in range(size):
                for a in neighbours_lattice[x, y]:
                    game = [[x, y], a]
                    game.sort()
                    game_list.append(game)
                    game_set.append(((game[0][0], game[0][1]), (game[1][0], game[1][1])))
                game_list.append([[x, y], [x, y]])
    return (game_list, game_set)


cdef str graph(game_set):
    """ Dump the neighbourhood to a graphviz dot file.
    """
    cdef str graph_string = "# dot graph..\n"
    graph_string += "\ngraph g {"
    graph_string += "\nnode [shape=point];"
    graph_string += "\nedge [weight=0.1];"
    for a, b in set(game_set):
        graph_string += "\n\"%do%d\" [pos=\"%d, %d\"];" % (a[0], a[1], a[0], a[1])
        graph_string += "\n\"%do%d\" -- \"%do%d\";" % (a[0], a[1], b[0], b[1])
    graph_string += "\n}\n"
    return graph_string


cdef unsigned int size = 5
cdef unsigned int num_games
cdef unsigned int toroid
cdef unsigned int scale
cdef float dmin
cdef float dmax
cdef float d


@cython.boundscheck(False)
def __main__():
    global num_games, toroid, d, dmin, dmax, size, scale

    # Check parameters
    if len(sys.argv) < 11:
        print("There are 10 required parameters:\nbmin & bmax - positive real numbers indicating the range of b values to test.\n" +
        "size - the size of the world, which will be a size*size square\n" +
        "dump_graph - 1 to dump graph stats and a .dot file for each small world.\n" +
        "gui_on - 1 to show a gui, 0 to not.\n" +
        "max_rounds - number of rounds to play for on each trial.\n" +
        "brange_num - number of values between bmin & bmax to use.\n" +
        "rewirings - number of randomly rewired small world graphs to test on each trial.\n" +
        "start_states - number of random start states to use.\n" +
        "file_prefix - file prefix to use for dumping stats & graphs.\n")
        return None

    # Collect parameters

    # Range of temptations to defect
    dmin = float(sys.argv[1])
    dmax = float(sys.argv[2])
    # Width & height of world
    size = int(sys.argv[3])
    # Whether to output a graph of the world
    cdef int dump_graph = int(sys.argv[4])
    # Whether to show the gui
    cdef int gui_on = int(sys.argv[5])
    # Maximum number of rounds to play for
    cdef int max_rounds = int(sys.argv[6])
    # Number of values in the temptatation range
    cdef int drange_num = int(sys.argv[7])
    # Number of randomly rewired graphs to run
    cdef int rewirings = int(sys.argv[8])
    # Number of random start states to run
    cdef int start_states = int(sys.argv[9])
    # Log & graph file name
    cdef str file_prefix = str(sys.argv[10])

    cdef np.ndarray[DTYPE_f] drange = np.linspace(dmin, dmax, num=drange_num)

    cdef unsigned int i = 0
    scale = 5
    cdef unsigned int x, y, count_coop
    cdef float cooperators
    cdef str fill, label
    cdef np.ndarray[DTYPE_t, ndim = 2] player_lattice, start_player_lattice
    cdef np.ndarray[DTYPE_t, ndim = 2] last_player_lattice = np.ones((size, size), dtype=int)
    #player_lattice[1, 1] = 0
    cdef np.ndarray[DTYPE_f, ndim = 2] scores_lattice = np.zeros((size, size), dtype=float)
    cdef np.ndarray[DTYPE_t, ndim = 2] rectangles = np.zeros((size, size), dtype=int)

    if gui_on == 1:
        w = init_gui(rectangles)

    cdef long t_start

    cdef np.ndarray[DTYPE_f] cooperators_list = np.zeros((max_rounds), dtype=float)
    cdef np.ndarray[DTYPE_f] small_cooperators_list = np.zeros((max_rounds), dtype=float)

    cdef np.ndarray neighbours_lattice, small_neighbours_lattice
    cdef np.ndarray[DTYPE_t, ndim = 3] games
    cdef str logstring, graph_file_name

    cdef float pop_size = size ** 2
    # Make population relative sizes
    cdef np.ndarray[DTYPE_t] population = np.array([1] * int(pop_size * 0.6) + [0] * int(pop_size * 0.4))

    # Dump file prefix
    logfile = open(file_prefix + ".tsv", 'w')
    logfile.write("Round\tcooperators\tc\tstart_state\tb\tsmw_id")
    logfile.close()

    cdef int c_count = 0
    cdef int b_count = 0

    # Various degrees of smallness
    for c in np.linspace(0.0, 0.1, 10):
        neighbours_base = [[neighbours(x, y) for y in range(size)] for x in range(size)]
        neighbours_lattice = np.array(neighbours_base)
        #small_worldify(neighbours_base, c)
        small_neighbours_lattice = np.array(neighbours_base)
        game_list, game_set = make_games(neighbours_lattice)
        small_game_list, small_game_set = make_games(small_neighbours_lattice)

        games = np.array(game_list, dtype=int)
        small_games = np.array(small_game_list, dtype=int)
        num_games = len(games)
        # For various temptation values
        for b in drange:
            # For several random start states
            for s in range(start_states):
                np.random.shuffle(population)
                start_player_lattice = population.reshape((size, size))
                # start_player_lattice[size / 2, size / 2] = 0
                # Run several random rewirings of the small world
                for r in range(rewirings):
                    player_lattice = start_player_lattice.copy()
                    neighbours_base = [[neighbours(x, y) for y in range(size)] for x in range(size)]

                    if c > 0:
                        small_worldify(neighbours_base, c)

                    small_neighbours_lattice = np.array(neighbours_base)

                    small_game_list, small_game_set = make_games(small_neighbours_lattice)

                    if dump_graph == 1:
                        graph_file_name = "%s_graph_%d_%f.dot" % (file_prefix, size, b)
                        graph_file = open(graph_file_name, 'w')
                        graph_file.write(graph(small_game_set))
                        graph_file.close()

                    small_games = np.array(small_game_list, dtype=int)
                    # Run the small world
                    for i in range(max_rounds):
                        #t_start = time.time()
                        if gui_on == 1:
                            update_gui(w, rectangles, player_lattice, last_player_lattice)

                        count_coop = num_coop(player_lattice)
                        cooperators = count_coop / float(size ** 2)
                        small_cooperators_list[i] += cooperators
                        # Round\tcooperators\tc\tstart_state\tb\tsmw_id
                        logstring = "\n%d\t%f\t%f\t%d-%d-%d\t%f\t%d_%d_%d_%d" % (i, cooperators, c, s, c_count, b_count, b, c_count, r, s, b_count)
                        print logstring
                        logfile = open(file_prefix + ".tsv", 'a')
                        logfile.write(logstring)
                        logfile.close()
                        #t_start = time.time()
                        if count_coop > 0:
                            play_round(player_lattice, scores_lattice, small_neighbours_lattice, b, last_player_lattice, small_games)
                        scores_lattice = np.zeros((size, size), dtype=float)

            b_count += 1

            cooperators_list = np.zeros((max_rounds), dtype=float)
            small_cooperators_list = np.zeros((max_rounds), dtype=float)
        c_count += 1
    logfile.close()
    if gui_on == 1:
        master.mainloop()
