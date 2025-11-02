
# Tic Tac Toe mit Minimax und Alpha-Beta-Pruning

# MAX = Spieler X (beginnt, will +1 erreichen)
# MIN = Spieler O (will -1 erreichen)
# Utility:
# +1 X gewinnt
# -1 O gewinnt
#  0 Unentschieden


# Grundlegende Spiellogik

X, O, EMPTY = "X", "O", " "
INF = 10**9  # sehr großer Wert für Minimax-Vergleich

# mögliche Gewinnlinien
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # Reihen
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # Spalten
    (0, 4, 8), (2, 4, 6)               # Diagonal
]

def initial_state():
    """Erzeugt ein leeres Spielfeld (Liste mit 9 Feldern)"""
    return [EMPTY] * 9

def player(board):
    """Gibt zurück, wer am Zug ist: X beginnt, dann abwechselnd"""
    x = sum(v == X for v in board)
    o = sum(v == O for v in board)
    return X if x == o else O

def actions(board):
    """Gibt eine Liste aller freien Felder (Index 0–8) zurück"""
    return [i for i, v in enumerate(board) if v == EMPTY]

def result(board, action):
    """Wendet einen Zug auf das Brett an und gibt ein neues Brett zurück"""
    if board[action] != EMPTY:
        raise ValueError("Ungültiger Zug")
    new_board = board.copy()
    new_board[action] = player(board)
    return new_board

def winner(board):
    """Überprüft, ob jemand gewonnen hat (X oder O)"""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None

def terminal(board):
    """Prüft, ob das Spiel vorbei ist (Sieg oder volles Brett)"""
    return winner(board) is not None or all(v != EMPTY for v in board)

def utility(board):
    """Bewertet den Endzustand (aus Sicht von X = MAX)"""
    w = winner(board)
    if w == X:
        return 1
    elif w == O:
        return -1
    else:
        return 0

# Reihenfolge für die Züge Mitte > Ecken > Kanten
def ordered_actions(board):
    """Sortiert mögliche Züge nach Wichtigkeit (Mitte zuerst)"""
    priority = {4: 3, 0: 2, 2: 2, 6: 2, 8: 2, 1: 1, 3: 1, 5: 1, 7: 1}
    return sorted(actions(board), key=lambda i: -priority[i])

# ---------- 1) Minimax-Algorithmus ----------

def minimax_best_action(board, use_order=False, count=None):
    """Berechnet den besten Zug mit Minimax (wie in der Vorlesung)"""
    if count is None:
        count = {'nodes': 0}  # Zähler für besuchte Knoten

    def max_value(state):
        """MAX-Knoten (X) – versucht, den Wert zu maximieren"""
        count['nodes'] += 1
        if terminal(state):
            return utility(state)
        v = -INF
        it = ordered_actions(state) if use_order else actions(state)
        for a in it:
            v = max(v, min_value(result(state, a)))
        return v

    def min_value(state):
        """MIN-Knoten (O) – versucht, den Wert zu minimieren"""
        count['nodes'] += 1
        if terminal(state):
            return utility(state)
        v = INF
        it = ordered_actions(state) if use_order else actions(state)
        for a in it:
            v = min(v, max_value(result(state, a)))
        return v

    # Wurzelknoten hängt davon ab, wer gerade dran ist
    p = player(board)
    it = ordered_actions(board) if use_order else actions(board)
    best_action = None

    if p == X:  # MAX am Zug
        best_value = -INF
        for a in it:
            v = min_value(result(board, a))
            if v > best_value:
                best_value, best_action = v, a
    else:       # MIN am Zug
        best_value = INF
        for a in it:
            v = max_value(result(board, a))
            if v < best_value:
                best_value, best_action = v, a

    return best_action, count['nodes']

# 2) Alpha-Beta-Pruning

def alphabeta_best_action(board, use_order=False, count=None):
    """Berechnet den besten Zug mit Alpha-Beta-Pruning"""
    if count is None:
        count = {'nodes': 0}  # Zähler für Knoten

    def max_value(state, alpha, beta):
        """MAX-Knoten (X) mit Alpha-Beta-Pruning"""
        count['nodes'] += 1
        if terminal(state):
            return utility(state)
        v = -INF
        it = ordered_actions(state) if use_order else actions(state)
        for a in it:
            v = max(v, min_value(result(state, a), alpha, beta))
            alpha = max(alpha, v)
            if beta <= alpha:  # Pruning-Bedingung
                break
        return v

    def min_value(state, alpha, beta):
        """MIN-Knoten (O) mit Alpha-Beta-Pruning"""
        count['nodes'] += 1
        if terminal(state):
            return utility(state)
        v = INF
        it = ordered_actions(state) if use_order else actions(state)
        for a in it:
            v = min(v, max_value(result(state, a), alpha, beta))
            beta = min(beta, v)
            if beta <= alpha:  # Pruning-Bedingung
                break
        return v

    # Wurzelknoten startet mit Alpha=-∞, Beta=+∞
    p = player(board)
    alpha, beta = -INF, INF
    it = ordered_actions(board) if use_order else actions(board)
    best_action = None

    if p == X:  # MAX am Zug
        best_value = -INF
        for a in it:
            v = min_value(result(board, a), alpha, beta)
            if v > best_value:
                best_value, best_action = v, a
            alpha = max(alpha, best_value)
    else:       # MIN am Zug
        best_value = INF
        for a in it:
            v = max_value(result(board, a), alpha, beta)
            if v < best_value:
                best_value, best_action = v, a
            beta = min(beta, best_value)

    return best_action, count['nodes']

# 3) Vergleich der Knoten

def benchmark():
    """Vergleicht Minimax und Alpha-Beta anhand von Knotenanzahl"""
    start = initial_state()

    # A) Startstellung
    a_mm, n_mm = minimax_best_action(start, use_order=False, count={'nodes': 0})
    a_ab, n_ab = alphabeta_best_action(start, use_order=False, count={'nodes': 0})
    a_mmo, n_mmo = minimax_best_action(start, use_order=True, count={'nodes': 0})
    a_abo, n_abo = alphabeta_best_action(start, use_order=True, count={'nodes': 0})

    # B) Mittelspiel
    mid = [X, O, X,
           O, X, EMPTY,
           EMPTY, EMPTY, O]  # X ist am Zug
    m_mm, mn_mm = minimax_best_action(mid, use_order=False, count={'nodes': 0})
    m_ab, mn_ab = alphabeta_best_action(mid, use_order=False, count={'nodes': 0})
    m_mmo, mn_mmo = minimax_best_action(mid, use_order=True, count={'nodes': 0})
    m_abo, mn_abo = alphabeta_best_action(mid, use_order=True, count={'nodes': 0})

    # Ausgabe der Ergebnisse
    print("Vergleich Minimax vs. Alpha-Beta (Knoten)\n")
    print("Start (ohne Ordering):")
    print(f"  Minimax:   Zug = {a_mm},   Knoten = {n_mm}")
    print(f"  AlphaBeta: Zug = {a_ab},   Knoten = {n_ab}\n")

    print("Start (mit Ordering):")
    print(f"  Minimax:   Zug = {a_mmo},  Knoten = {n_mmo}")
    print(f"  AlphaBeta: Zug = {a_abo},  Knoten = {n_abo}\n")

    print("Mittelspiel (ohne Ordering):")
    print(f"  Minimax:   Zug = {m_mm},   Knoten = {mn_mm}")
    print(f"  AlphaBeta: Zug = {m_ab},   Knoten = {mn_ab}\n")

    print("Mittelspiel (mit Ordering):")
    print(f"  Minimax:   Zug = {m_mmo},  Knoten = {mn_mmo}")
    print(f"  AlphaBeta: Zug = {m_abo},  Knoten = {mn_abo}")

# Programmstart
if __name__ == "__main__":
    benchmark()
