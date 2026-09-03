"""Launch a command in its own session so it survives the parent's teardown."""
import os, sys
log = sys.argv[1]; cmd = sys.argv[2:]
if os.fork(): os._exit(0)
os.setsid()
if os.fork(): os._exit(0)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.close(os.open(os.devnull, os.O_RDONLY))
os.execvp(cmd[0], cmd)
