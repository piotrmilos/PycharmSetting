echo 'compiling with cython...';
cython3 -3 util.pyx -o util.c;
cython3 -3 game/reversi.pyx -o game/reversi.c;
cython3 -3 game/board.pyx -o game/board.c;
cython3 -3 agents/monte_carlo_agent.pyx -o agents/monte_carlo_agent.c;

echo 'removing old .so files...';
rm *.so agents/*.so game/*.so;

echo 'compiling .c files with gcc...'
gcc -shared -I/usr/include/python3.4m -fPIC -pthread -fwrapv -O3 -Wall -fno-strict-aliasing util.c -o util.so; 
gcc -shared -I/usr/include/python3.4m -fPIC -pthread -fwrapv -O3 -Wall -fno-strict-aliasing game/reversi.c -o game/reversi.so;
gcc -shared -I/usr/include/python3.4m -fPIC -pthread -fwrapv -O3 -Wall -fno-strict-aliasing game/board.c -o game/board.so; 
gcc -shared -I/usr/include/python3.4m -fPIC -pthread -fwrapv -O3 -Wall -fno-strict-aliasing agents/monte_carlo_agent.c -o agents/monte_carlo_agent.so; 

echo 'removing .c files...';
rm *.c agents/*.c game/*.c;

echo 'done.';

