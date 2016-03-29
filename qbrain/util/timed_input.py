__author__ = 'Marek'
import sys
import select


def timed_unix_input(message, seconds):
    print(message)
    i, o, e = select.select([sys.stdin], [], [], seconds)

    if i:
        return sys.stdin.readline().strip().decode()
    else:
        return None
