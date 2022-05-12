[solver]
km = 0.6
kr =  0.6
use pop mult = False
pop mult = 3
pop size = 50
use max generations = False
max generations = 500
max generation mult = 6
use start guess = True
use boundaries = False
sleep time = 0.5
max log elements = 100000
use parallel processing = False
parallel processes = 2
parallel chunksize = 25
use autosave = False
save all evals = False
autosave interval = 10
create trial = best_1_bin
errorbar level = 1.05
allowed fom discrepancy = 1e-10
ignore fom nan = True
ignore fom inf = True

[data commands]
names = A Example;Default;Simulation;Sustematic Errors
x commands = x+33;x;arange(0.01, 6, 0.01);x
y commands = y/1e5;y;arange(0.01, 6, 0.01)*0;y
e commands = e/2.;e;arange(0.01, 6, 0.01)*0;rms(e, fpe(1.0, 0.02), 0.01*dydx())

[data handling]
toggle show = True
data loader = default

[parameters]
registred classes = Layer;Stack;Sample;Instrument;UserVars;Surface;Bulk
set func = set

[plugins]
loaded plugins = SimpleReflectivity;SimpleLayer

[data plot]
zoom = False
autoscale = True
y scale = log
x scale = lin

[fom plot]
zoom = False
autoscale = True
y scale = lin
x scale = lin

[pars plot]
zoom = False
autoscale = True
y scale = lin
x scale = lin

[fom scan plot]
zoom = False
autoscale = True
y scale = lin
x scale = lin

[sample plot]
zoom = False
autoscale = True
y scale = lin
x scale = lin
data derived color = False
show single model = False
legend outside = False
show imag = False

[startup]
show profiles = True
widescreen = True
