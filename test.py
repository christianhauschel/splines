from subprocess import run


run("python -m unittest test_splines.py", cwd="tests/")
run("python -m unittest test_tools.py", cwd="tests/")