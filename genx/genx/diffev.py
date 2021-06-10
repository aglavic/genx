'''File: diffev.py an implementation of the differential evolution algoithm
for fitting.
Programmed by: Matts Bjorck
Last changed: 2008 11 23
'''

from numpy import *
from .gui_logging import iprint
from logging import debug
import _thread
import time
import random as random_mod
import sys, os, pickle

__mpi_loaded__=False
__parallel_loaded__=False
_cpu_count=1

try:
    import multiprocessing as processing

    __parallel_loaded__=True
    _cpu_count=processing.cpu_count()
except ImportError:
    debug('processing not installed no parallel processing possible')
    processing=None

try:
    from mpi4py import MPI as mpi
except ImportError:
    rank=0
    __mpi_loaded__=False
    mpi=None
else:
    __mpi_loaded__=True
    comm=mpi.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()

from . import model as mmodel

from .lib.Simplex import Simplex

# Add current path to the system paths
# just in case some user make a directory change
sys.path.append(os.getcwd())

# ==============================================================================
# class: DiffEv
class DiffEv:
    '''
    Class DiffEv
    Contains the implemenetation of the differential evolution algorithm.
    It also contains thread support which is activated by the start_fit 
    function.
    '''

    export_parameters={'km': float, 'kr': float, 'pf': float, 'use_pop_mult': bool, 'pop_mult': int, 'pop_size': int,
                       'use_max_generations': bool, 'max_generations': int, 'max_generation_mult': int,
                       'use_start_guess': bool, 'use_boundaries': bool, 'sleep_time': float,
                       'use_parallel_processing': bool, 'use_mpi': bool, 'fom_log': array, 'start_guess': array,
                       'limit_fit_range': bool, 'fit_xmin': float, 'fit_xmax': float,
                       }

    parameter_groups=[
        ['Fitting', ['use_start_guess', 'use_boundaries', 'use_autosave', 'autosave_interval']],
        ['Diff. Ev.', ['km', 'kr', 'method']],
        ['Population size', ['use_pop_mult', 'pop_mult', 'pop_size']],
        ['Max. Generatrions', ['use_max_generations', 'max_generations', 'max_generation_mult']],
        ['Parallel processing', ['use_parallel_processing', 'processes', 'chunksize']],
        ]

    def create_mutation_table(self):
        # Mutation schemes implemented
        self.mutation_schemes=[self.best_1_bin, self.rand_1_bin,
                               self.best_either_or, self.rand_either_or, self.jade_best, self.simplex_best_1_bin]

    def __init__(self):
        self.create_mutation_table()

        self.model=mmodel.Model()

        self.km=0.7  # Mutation constant
        self.kr=0.7  # Cross over constant
        self.pf=0.5  # probablility for mutation
        self.c=0.07
        self.simplex_interval=5  # Interval of running the simplex opt
        self.simplex_step=0.05  # first step as a fraction of pop size
        self.simplex_n=0.0  # Number of individuals that will be optimized by simplex
        self.simplex_rel_epsilon=1000  # The relative epsilon - convergence critera
        self.simplex_max_iter=100  # THe maximum number of simplex runs
        # Flag to choose beween the two alternatives below
        self.use_pop_mult=False
        self.pop_mult=3  # Set the pop_size to pop_mult * # free parameters
        self.pop_size=10  # Set the pop_size only

        # Flag to choose between the two alternatives below
        self.use_max_generations=False
        self.max_generations=500  # Use a fixed # of iterations
        self.max_generation_mult=6  # A mult const for max number of iter

        # Flag to choose whether or not to use a starting guess
        self.use_start_guess=True
        # Flag to choose wheter or not to use the boundaries
        self.use_boundaries=True

        # Sleeping time for every generation
        self.sleep_time=0.2
        # Allowed disagreement between the two different fom
        # evaluations
        self.fom_allowed_dis=1e-10
        # Flag if we should use parallel processing 
        self.use_parallel_processing=__parallel_loaded__*0
        if __parallel_loaded__:
            self.processes=_cpu_count
        else:
            self.processes=0

        self.chunksize=1

        # Flag for using mpi
        self.use_mpi=False

        # Flag for using autosave
        self.use_autosave=True
        # autosave interval in generations        
        self.autosave_interval=10

        # Functions that are user definable
        self.plot_output=default_plot_output
        self.text_output=default_text_output
        self.parameter_output=default_parameter_output
        self.autosave=defualt_autosave
        self.fitting_ended=default_fitting_ended

        # Definition for the create_trial function
        self.create_trial=self.best_1_bin
        self.update_pop=self.standard_update_pop
        self.init_new_generation=self.standard_init_new_generation

        # Control flags:
        self.running=False  # true if optimization is running
        self.stop=False  # true if the optimization should stop
        self.setup_ok=False  # True if the optimization have been setup
        self.error=False  # True/string if an error ahs occured

        # Logging variables
        # Maximum number of logged elements
        self.max_log=100000
        self.fom_log=array([[0, 0]])[0:0]
        # self.par_evals = array([[]])[0:0]

        self.par_evals=CircBuffer(self.max_log, buffer=array([[]])[0:0])
        # self.fom_evals = array([])
        self.fom_evals=CircBuffer(self.max_log)

        self.start_guess=array([])

        self.limit_fit_range=False
        self.fit_xmin=0.01
        self.fit_xmax=0.1

    @property
    def method(self):
        return self.create_trial.__name__

    @method.setter
    def method(self, value):
        names=self.methods
        if value in names:
            self.create_trial=self.mutation_schemes[names.index(value)]
        else:
            raise ValueError("Mutation method has to be in %s"%names)

    @property
    def methods(self):
        return [f.__name__ for f in self.mutation_schemes]

    def write_h5group(self, group, clear_evals=False):
        """ Write parameters into hdf5 group

        :param group: h5py Group to write into
        :return:
        """
        for par in self.export_parameters:
            obj=getattr(self, par)
            group[par]=obj

        if clear_evals:
            group['par_evals']=self.par_evals.array()[0:0]
            group['fom_evals']=self.fom_evals.array()[0:0]
        else:
            group['par_evals']=self.par_evals.array()
            group['fom_evals']=self.fom_evals.array()

    def read_h5group(self, group):
        """ Read parameters from a hdf5 group

        :param group: h5py Group to read from
        :return:
        """
        self.setup_ok=False
        for par in self.export_parameters:
            obj=getattr(self, par)
            if self.export_parameters[par] is array:
                setattr(self, par, group[par][()])
            else:
                try:
                    setattr(self, par, self.export_parameters[par](group[par][()]))
                except KeyError:
                    iprint("No value found for %s"%par)
                    continue

        self.par_evals.copy_from(group['par_evals'][()])
        self.fom_evals.copy_from(group['fom_evals'][()])

    def safe_copy(self, object):
        '''safe_copy(self, object) --> None
        
        Does a safe copy of object to this object. Makes copies of everything 
        if necessary. The two objects become decoupled.
        '''
        self.km=object.km  # Mutation constant
        self.kr=object.kr  # Cross over constant
        self.pf=object.pf  # probablility for mutation

        # Flag to choose beween the two alternatives below
        self.use_pop_mult=object.use_pop_mult
        self.pop_mult=object.pop_mult
        self.pop_size=object.pop_size

        # Flag to choose between the two alternatives below
        self.use_max_generations=object.use_max_generations
        self.max_generations=object.max_generations
        self.max_generation_mult=object.max_generation_mult

        # Flag to choose whether or not to use a starting guess
        self.use_start_guess=object.use_start_guess
        # Flag to choose wheter or not to use the boundaries
        self.use_boundaries=object.use_boundaries

        # Sleeping time for every generation
        self.sleep_time=object.sleep_time
        # Flag if we should use parallel processing 
        if __parallel_loaded__:
            self.use_parallel_processing=object.use_parallel_processing
        else:
            self.use_parallel_processing=False

        print("copy")
        try:
            self.limit_fit_range=object.limit_fit_range
            self.fit_xmin=object.fit_xmin
            self.fit_xmax=object.fit_xmax
        except AttributeError:
            self.limit_fit_range=False

        # Flag if we should use mpi
        if __mpi_loaded__:
            try:
                self.use_mpi=object.use_mpi
            except AttributeError:
                self.use_mpi=False
        else:
            self.use_mpi=False

        # Definition for the create_trial function
        # self.set_create_trial(object.get_create_trial())

        # True if the optimization have been setup
        self.setup_ok=object.setup_ok

        # Logging variables
        self.fom_log=object.fom_log[:]
        self.par_evals.copy_from(object.par_evals)
        self.fom_evals.copy_from(object.fom_evals)

        if self.setup_ok:
            self.n_pop=object.n_pop
            self.max_gen=object.max_gen

            # Starting values setup
            self.pop_vec=object.pop_vec

            self.start_guess=object.start_guess

            self.trial_vec=object.trial_vec
            self.best_vec=object.best_vec

            self.fom_vec=object.fom_vec
            self.best_fom=object.best_fom
            # Not all implementaions has these copied within their files
            # Just ignore if an error occur
            try:
                self.n_dim=object.n_dim
                self.par_min=object.par_min
                self.par_max=object.par_max
            except:
                pass

    def pickle_string(self, clear_evals=False):
        '''Pickle the object.

        Saves a copy into a pickled string note that the dynamic
        functions will not be saved. For normal use this is taken care of
        outside this class with the config object.
        '''
        cpy=DiffEv()
        cpy.safe_copy(self)
        if clear_evals:
            cpy.par_evals.buffer=cpy.par_evals.buffer[0:0]
            cpy.fom_evals.buffer=cpy.fom_evals.buffer[0:0]
        cpy.create_trial=None
        cpy.update_pop=None
        cpy.init_new_generation=None
        cpy.plot_output=None
        cpy.text_output=None
        cpy.parameter_output=None
        cpy.autosaves=None
        cpy.fitting_ended=None
        cpy.model=None
        cpy.mutation_schemes=None

        return pickle.dumps(cpy)

    def pickle_load(self, pickled_string):
        '''load_pickles(self, pickled_string) --> None
        
        Loads the pickled string into the this object. See pickle_string.
        '''
        obj=pickle.loads(pickled_string, encoding='latin1', errors='ignore')
        obj.create_mutation_table()
        self.safe_copy(obj)

    def reset(self):
        ''' reset(self) --> None
        
        Resets the optimizer. Note this has to be run if the optimizer is to
        be restarted.
        '''
        self.setup_ok=False

    def connect_model(self, model):
        '''connect_model(self, model) --> None
        
        Connects the model [model] to this object. Retrives the function
        that sets the variables  and stores a reference to the model.
        '''
        # Retrive parameters from the model
        (par_funcs, start_guess, par_min, par_max)=model.get_fit_pars()

        # Control parameter setup
        self.par_min=array(par_min)
        self.par_max=array(par_max)
        self.par_funcs=par_funcs
        self.model=model
        self.n_dim=len(par_funcs)
        model.limit_fit_range, model.fit_xmin, model.fit_xmax=(
            self.limit_fit_range,
            self.fit_xmin,
            self.fit_xmax)
        if not self.setup_ok:
            self.start_guess=start_guess

    def init_fitting(self, model):
        '''
        Function to run before a new fit is started with start_fit.
        It initilaize the population and sets the limits on the number
        of generation and the population size.
        '''
        self.connect_model(model)
        if self.use_pop_mult:
            self.n_pop=int(self.pop_mult*self.n_dim)
        else:
            self.n_pop=int(self.pop_size)
        if self.use_max_generations:
            self.max_gen=int(self.max_generations)
        else:
            self.max_gen=int(self.max_generation_mult*self.n_dim*self.n_pop)

        # Starting values setup
        self.pop_vec=[self.par_min+random.rand(self.n_dim)*(self.par_max-self.par_min)
                      for i in range(self.n_pop)]

        if self.use_start_guess:
            self.pop_vec[0]=array(self.start_guess)

        self.trial_vec=[zeros(self.n_dim) for i in range(self.n_pop)]
        self.best_vec=self.pop_vec[0]

        self.fom_vec=zeros(self.n_dim)
        self.best_fom=1e20

        # Storage area for JADE archives
        self.km_vec=ones(self.n_dim)*self.km
        self.kr_vec=ones(self.n_dim)*self.kr

        # Logging varaibles
        self.fom_log=array([[0, 1]])[0:0]
        self.par_evals=CircBuffer(self.max_log, buffer=array([self.par_min])[0:0])
        # self.fom_evals = array([])
        self.fom_evals=CircBuffer(self.max_log)
        # Number of FOM evaluations
        self.n_fom=0
        # self.par_evals.reset(array([self.par_min])[0:0])
        # self.fom_evals.reset()

        if rank==0:
            self.text_output('DE initilized')

        # Remeber that everything has been setup ok
        self.setup_ok=True

    def init_fom_eval(self):
        '''init_fom_eval(self) --> None
        
        Makes the eval_fom function
        '''
        model=self.model

        # Setting up for parallel processing
        if self.use_parallel_processing and __parallel_loaded__:
            self.text_output('Setting up a pool of workers ...')
            self.setup_parallel()
            self.eval_fom=self.calc_trial_fom_parallel
        elif self.use_mpi and __mpi_loaded__:
            self.setup_parallel_mpi()
            self.eval_fom=self.calc_trial_fom_parallel_mpi
        else:
            self.eval_fom=self.calc_trial_fom

    def start_fit(self, model):
        '''
        Starts fitting in a seperate thred.
        '''
        # If it is not already running
        if not self.running:
            # Initilize the parameters to fit
            self.reset()
            self.init_fitting(model)
            self.init_fom_eval()
            self.stop=False
            # Start fitting in a new thread
            _thread.start_new_thread(self.optimize, ())
            # For debugging
            # self.optimize()
            self.text_output('Starting the fit...')
            # self.running = True
            return True
        else:
            self.text_output('Fit is already running, stop and then start')
            return False

    def stop_fit(self):
        '''
        Stops the fit if it has been started in a seperate theres 
        by start_fit.
        '''
        # If not running stop
        if self.running:
            self.stop=True
            self.text_output('Trying to stop the fit...')
        else:
            self.text_output('The fit is not running')

    def resume_fit(self, model):
        '''
        Resumes the fitting if has been stopped with stop_fit.
        '''
        if not self.running:
            self.stop=False
            self.connect_model(model)
            self.init_fom_eval()
            n_dim_old=self.n_dim
            if self.n_dim==n_dim_old:
                _thread.start_new_thread(self.optimize, ())
                self.text_output('Restarting the fit...')
                self.running=True
                return True
            else:
                self.text_output('The number of parameters has changed'
                                 ' restart the fit.')
                return False
        else:
            self.text_output('Fit is already running, stop and then start')
            return False

    def optimize(self):
        """Method that does the optimization.

        Note that this method does not run in a separate thread.
        For threading use start_fit, stop_fit and resume_fit instead.
        """
        if self.use_mpi:
            self.optimize_mpi()
        else:
            self.optimize_standard()

    def optimize_standard(self):
        '''
        Method implementing the main loop of the differential evolution
        algorithm. Note that this method does not run in a separate thread.
        For threading use start_fit, stop_fit and resume_fit instead.
        '''

        self.text_output('Calculating start FOM ...')
        self.running=True
        self.error=False
        self.n_fom=0
        ## Old leftovers before going parallel
        # self.fom_vec = [self.calc_fom(vec) for vec in self.pop_vec]
        # [self.par_evals.append(vec, axis = 0) for vec in self.pop_vec]
        # [self.fom_evals.append(vec) for vec in self.fom_vec]
        # New parallel calcualtions
        self.trial_vec=self.pop_vec[:]
        self.eval_fom()
        [self.par_evals.append(vec, axis=0) for vec in self.pop_vec]
        [self.fom_evals.append(vec) for vec in self.trial_fom]
        self.fom_vec=self.trial_fom[:]
        # print self.fom_vec
        best_index=argmin(self.fom_vec)
        # print self.fom_vec
        # print best_index
        self.best_vec=copy(self.pop_vec[best_index])
        # print self.best_vec
        self.best_fom=self.fom_vec[best_index]
        # print self.best_fom
        if len(self.fom_log)==0:
            self.fom_log=r_[self.fom_log, \
                            [[len(self.fom_log), self.best_fom]]]
        # Flag to keep track if there has been any improvemnts
        # in the fit - used for updates
        self.new_best=True

        self.text_output('Going into optimization ...')

        # Update the plot data for any gui or other output
        self.plot_output(self)
        self.parameter_output(self)

        # Just making gen live in this scope as well...
        gen=self.fom_log[-1, 0]
        for gen in range(int(self.fom_log[-1, 0])+1, self.max_gen+int(self.fom_log[-1, 0])+1):
            if self.stop:
                break

            t_start=time.time()

            self.init_new_generation(gen)

            # Create the vectors who will be compared to the 
            # population vectors
            [self.create_trial(index) for index in range(self.n_pop)]
            self.eval_fom()
            # Calculate the fom of the trial vectors and update the population
            [self.update_pop(index) for index in range(self.n_pop)]

            # Add the evaluation to the logging
            [self.par_evals.append(vec, axis=0) for vec in self.trial_vec]
            [self.fom_evals.append(vec) for vec in self.trial_fom]

            # Add the best value to the fom log
            self.fom_log=r_[self.fom_log, [[len(self.fom_log), self.best_fom]]]

            # Let the model calculate the simulation of the best.
            sim_fom=self.calc_sim(self.best_vec)

            # Sanity of the model does the simualtions fom agree with
            # the best fom
            if abs(sim_fom-self.best_fom)>self.fom_allowed_dis:
                self.text_output('Disagrement between two different fom'
                                 ' evaluations')
                self.error=('The disagreement between two subsequent '
                            'evaluations is larger than %s. Check the '
                            'model for circular assignments.'
                            %self.fom_allowed_dis)
                break

            # Update the plot data for any gui or other output
            self.plot_output(self)
            self.parameter_output(self)

            # Let the optimization sleep for a while
            if self.use_parallel_processing:
                # limit the length of each iteration in parallel processing
                # at least on windows there is no issue with fast iterations in single thread
                to_sleep=self.sleep_time-(time.time()-t_start)
                if to_sleep>0:
                    time.sleep(to_sleep)

            # Time measurent to track the speed
            t=time.time()-t_start
            if t>0:
                speed=self.n_pop/t
            else:
                speed=999999
            self.text_output('FOM: %.3f Generation: %d Speed: %.1f'% \
                             (self.best_fom, gen, speed))

            self.new_best=False
            # Do an autosave if activated and the interval is coorect
            if gen%self.autosave_interval==0 and self.use_autosave:
                self.autosave()

        if not self.error:
            self.text_output('Stopped at Generation: %d after %d fom evaluations...'%(gen, gen*self.n_pop))

        # Lets clean up and delete our pool of workers
        if self.use_parallel_processing:
            self.dismount_parallel()
        self.eval_fom=None

        # Now the optimization has stopped
        self.running=False

        # Run application specific clean-up actions
        self.fitting_ended(self)

    def optimize_mpi(self):
        '''
        Method implementing the main loop of the differential evolution
        algorithm using mpi. This should only be used from the command line.
        The gui can not handle to use mpi.
        '''

        if rank==0:
            self.text_output('Calculating start FOM ...')
        self.running=True
        self.error=False
        self.n_fom=0
        # Old leftovers before going parallel
        self.fom_vec=[self.calc_fom(vec) for vec in self.pop_vec]
        [self.par_evals.append(vec, axis=0) \
         for vec in self.pop_vec]
        [self.fom_evals.append(vec) for vec in self.fom_vec]
        # print self.fom_vec
        best_index=argmin(self.fom_vec)
        # print self.fom_vec
        # print best_index
        self.best_vec=copy(self.pop_vec[best_index])
        # print self.best_vec
        self.best_fom=self.fom_vec[best_index]
        # print self.best_fom
        if len(self.fom_log)==0:
            self.fom_log=r_[self.fom_log, \
                            [[len(self.fom_log), self.best_fom]]]
        # Flag to keep track if there has been any improvemnts
        # in the fit - used for updates
        self.new_best=True

        if rank==0:
            self.text_output('Going into optimization ...')

        # Update the plot data for any gui or other output
        self.plot_output(self)
        self.parameter_output(self)

        # Just making gen live in this scope as well...
        gen=self.fom_log[-1, 0]
        for gen in range(int(self.fom_log[-1, 0])+1, self.max_gen \
                                                     +int(self.fom_log[-1, 0])+1):
            if self.stop:
                break
            t_start=time.time()

            self.init_new_generation(gen)

            # Create the vectors who will be compared to the
            # population vectors
            if rank==0:
                [self.create_trial(index) for index in range(self.n_pop)]
                tmp_trial_vec=self.trial_vec
            else:
                tmp_trial_vec=0
            tmp_trial_vec=comm.bcast(tmp_trial_vec, root=0)
            self.trial_vec=tmp_trial_vec
            self.eval_fom()
            tmp_fom=self.trial_fom
            comm.Barrier()

            # collect foms and reshape them and set the completed tmp_fom to trial_fom
            tmp_fom=comm.gather(tmp_fom, root=0)
            if rank==0:
                tmp_fom_list=[]
                for i in list(tmp_fom):
                    tmp_fom_list=tmp_fom_list+i
                tmp_fom=tmp_fom_list

            tmp_fom=comm.bcast(tmp_fom, root=0)
            self.trial_fom=array(tmp_fom).reshape(self.n_pop, )

            [self.update_pop(index) for index in range(self.n_pop)]

            # Calculate the fom of the trial vectors and update the population
            if rank==0:
                # Add the evaluation to the logging
                [self.par_evals.append(vec, axis=0) for vec in self.trial_vec]
                [self.fom_evals.append(vec) for vec in self.trial_fom]

                # Add the best value to the fom log
                self.fom_log=r_[self.fom_log, [[len(self.fom_log), self.best_fom]]]

                # Let the model calculate the simulation of the best.
                sim_fom=self.calc_sim(self.best_vec)

                # Sanity of the model does the simualtions fom agree with
                # the best fom
                if abs(sim_fom-self.best_fom)>self.fom_allowed_dis and rank==0:
                    self.text_output('Disagrement between two different fom'
                                     ' evaluations')
                    self.error=('The disagreement between two subsequent '
                                'evaluations is larger than %s. Check the '
                                'model for circular assignments.'
                                %self.fom_allowed_dis)
                    break

                # Update the plot data for any gui or other output
                self.plot_output(self)
                self.parameter_output(self)

                # Let the optimization sleep for a while
                # time.sleep(self.sleep_time)

                # Time measurent to track the speed
                t=time.time()-t_start
                if t>0:
                    speed=self.n_pop/t
                else:
                    speed=999999
                if rank==0:
                    self.text_output('FOM: %.3f Generation: %d Speed: %.1f'%
                                     (self.best_fom, gen, speed))

                self.new_best=False
                # Do an autosave if activated and the interval is coorect
                if gen%self.autosave_interval==0 and self.use_autosave:
                    self.autosave()

        if rank==0:
            if not self.error:
                self.text_output('Stopped at Generation: %d after %d fom evaluations...'%(gen, gen*self.n_pop))

        # Lets clean up and delete our pool of workers

        self.eval_fom=None

        # Now the optimization has stopped
        self.running=False

        # Run application specific clean-up actions
        self.fitting_ended(self)

    def calc_fom(self, vec):
        '''
        Function to calcuate the figure of merit for parameter vector 
        vec.
        '''
        model=self.model
        model.limit_fit_range, model.fit_xmin, model.fit_xmax=(
            self.limit_fit_range,
            self.fit_xmin,
            self.fit_xmax)

        # Set the parameter values
        list(map(lambda func, value: func(value), self.par_funcs, vec))
        fom=self.model.evaluate_fit_func()
        self.n_fom+=1
        return fom

    def calc_trial_fom(self):
        '''
        Function to calculate the fom values for the trial vectors
        '''
        model=self.model
        model.limit_fit_range, model.fit_xmin, model.fit_xmax=(
            self.limit_fit_range,
            self.fit_xmin,
            self.fit_xmax)

        self.trial_fom=[self.calc_fom(vec) for vec in self.trial_vec]

    def calc_sim(self, vec):
        ''' calc_sim(self, vec) --> None
        Function that will evaluate the the data points for
        parameters in vec.
        '''
        model=self.model
        model.limit_fit_range, model.fit_xmin, model.fit_xmax=(
            self.limit_fit_range,
            self.fit_xmin,
            self.fit_xmax)
        # Set the paraemter values
        list(map(lambda func, value: func(value), self.par_funcs, vec))

        self.model.evaluate_sim_func()
        return self.model.fom

    def setup_parallel(self):
        '''setup_parallel(self) --> None
        
        setup for parallel proccesing. Creates a pool of workers with
        as many cpus there is available
        '''
        # check if CUDA has been activated
        from genx.models.lib import paratt
        use_cuda=paratt.Refl.__module__.rsplit('.',1)[1]=='paratt_cuda'
        self.pool=processing.Pool(processes=self.processes,
                                  initializer=parallel_init,
                                  initargs=(self.model.pickable_copy(), use_cuda))
        self.text_output("Starting a pool with %i workers ..."%(self.processes,))
        time.sleep(1.0)
        # print "Starting a pool with ", self.processes, " workers ..."

    def setup_parallel_mpi(self):
        """Inits the number or process used for mpi.
        """

        if rank==0:
            self.text_output("Inits mpi with %i processes ..."%(size,))
        parallel_init(self.model.pickable_copy())
        time.sleep(0.1)

    def dismount_parallel(self):
        ''' dismount_parallel(self) --> None
        Used to close the pool and all its processes
        '''
        self.pool.close()
        self.pool.join()

        # del self.pool

    def calc_trial_fom_parallel(self):
        '''calc_trial_fom_parallel(self) --> None
        
        Function to calculate the fom in parallel using the pool
        '''
        model=self.model
        model.limit_fit_range, model.fit_xmin, model.fit_xmax=(
            self.limit_fit_range,
            self.fit_xmin,
            self.fit_xmax)

        self.trial_fom=self.pool.map(parallel_calc_fom, self.trial_vec, chunksize=self.chunksize)

    def calc_trial_fom_parallel_mpi(self):
        """ Function to calculate the fom in parallel using mpi
        """
        model=self.model
        model.limit_fit_range, model.fit_xmin, model.fit_xmax=(
            self.limit_fit_range,
            self.fit_xmin,
            self.fit_xmax)

        step_len=int(len(self.trial_vec)/size)
        remainder=int(len(self.trial_vec)%size)
        left, right=0, 0

        if rank<=remainder-1:
            left=rank*(step_len+1)
            right=(rank+1)*(step_len+1)-1
        elif rank>remainder-1:
            left=remainder*(step_len+1)+(rank-remainder)*step_len
            right=remainder*(step_len+1)+(rank-remainder+1)*step_len-1
        fom_temp=[]

        for i in range(left, right+1):
            fom_temp.append(parallel_calc_fom(self.trial_vec[i]))

        self.trial_fom=fom_temp

    # noinspection PyArgumentList
    def calc_error_bar(self, index, fom_level):
        '''calc_error_bar(self, parameter) --> (error_bar_low, error_bar_high)
        
        Calculates the errorbar for one parameter number index. 
        returns a float tuple with the error bars. fom_level is the 
        level which is the upperboundary of the fom is allowed for the
        calculated error.
        '''
        # print self.par_evals.shape, self.par_evals
        # print self.fom_evals.shape, self.fom_evals
        if self.setup_ok:  # and len(self.par_evals) != 0:
            par_values=self.par_evals[:, index]
            # print (self.fom_evals < fom_level).sum()
            values_under_level=compress(self.fom_evals[:]<fom_level*self.best_fom, par_values)
            # print values_under_level
            error_bar_low=values_under_level.min()-self.best_vec[index]
            error_bar_high=values_under_level.max()-self.best_vec[index]
            return error_bar_low, error_bar_high
        else:
            raise ErrorBarError()

    def init_new_generation(self, gen):
        ''' Function that is called every time a new generation starts'''
        pass

    def standard_init_new_generation(self, gen):
        ''' Function that is called every time a new generation starts'''
        pass

    def standard_update_pop(self, index):
        '''
        Function to update population vector index. calcs the figure of merit
        and compares it to the current population vector and also checks
        if it is better than the current best.
        '''
        # fom = self.calc_fom(self.trial_vec[index])
        fom=self.trial_fom[index]
        if fom<self.fom_vec[index]:
            self.pop_vec[index]=self.trial_vec[index].copy()
            self.fom_vec[index]=fom
            if fom<self.best_fom:
                self.new_best=True
                self.best_vec=self.trial_vec[index].copy()
                self.best_fom=fom

    # noinspection PyArgumentList
    def simplex_old_init_new_generation(self, gen):
        '''It will run the simplex method every simplex_interval
             generation with a fracitonal step given by simple_step 
             on the best indivual as well a random fraction of simplex_n individuals.
        '''
        iprint('Inits new generation')
        if gen%self.simplex_interval==0:
            spread=array(self.trial_vec).max(0)-array(self.trial_vec).min(0)
            simp=Simplex(self.calc_fom, self.best_vec, spread*self.simplex_step)
            iprint('Starting simplex run for best vec')
            new_vec, err, iter=simp.minimize(epsilon=self.best_fom/self.simplex_rel_epsilon,
                                             maxiters=self.simplex_max_iter)
            iprint('FOM improvement: ', self.best_fom-err)

            if self.use_boundaries:
                # Check so that the parameters lie indside the bounds
                ok=bitwise_and(self.par_max>new_vec, self.par_min<new_vec)
                # If not inside make a random re-initilazation of that parameter
                new_vec=where(ok, new_vec, random.rand(self.n_dim)* \
                              (self.par_max-self.par_min)+self.par_min)

            new_fom=self.calc_fom(new_vec)
            if new_fom<self.best_fom:
                self.best_fom=new_fom
                self.best_vec=new_vec
                self.pop_vec[0]=new_vec
                self.fom_vec[0]=self.best_fom
                self.new_best=True

            # Apply the simplex to a simplex_n memebers (0-1)
            for index1 in random_mod.sample(range(len(self.pop_vec)),
                                            int(len(self.pop_vec)*self.simplex_n)):
                iprint('Starting simplex run for member: ', index1)
                mem=self.pop_vec[index1]
                mem_fom=self.fom_vec[index1]
                simp=Simplex(self.calc_fom, mem, spread*self.simplex_step)
                new_vec, err, iter=simp.minimize(epsilon=self.best_fom/self.simplex_rel_epsilon,
                                                 maxiters=self.simplex_max_iter)
                if self.use_boundaries:
                    # Check so that the parameters lie indside the bounds
                    ok=bitwise_and(self.par_max>new_vec, self.par_min<new_vec)
                    # If not inside make a random re-initilazation of that parameter
                    new_vec=where(ok, new_vec, random.rand(self.n_dim)* \
                                  (self.par_max-self.par_min)+self.par_min)

                new_fom=self.calc_fom(new_vec)
                if new_fom<mem_fom:
                    self.pop_vec[index1]=new_vec
                    self.fom_vec[index1]=new_fom
                    if new_fom<self.best_fom:
                        self.best_fom=new_fom
                        self.best_vec=new_vec
                        self.new_best=True

    # noinspection PyArgumentList
    def simplex_init_new_generation(self, gen):
        '''It will run the simplex method every simplex_interval
             generation with a fracitonal step given by simple_step 
             on the simplex_n*n_pop best individuals.
        '''
        iprint('Inits new generation')
        if gen%self.simplex_interval==0:
            spread=array(self.trial_vec).max(0)-array(self.trial_vec).min(0)

            indices=argsort(self.fom_vec)
            n_ind=int(self.n_pop*self.simplex_n)
            if n_ind==0:
                n_ind=1
            # Apply the simplex to a simplex_n memebers (0-1)
            for index1 in indices[:n_ind]:
                self.text_output('Starting simplex run for member: %d'%index1)
                mem=self.pop_vec[index1].copy()
                mem_fom=self.fom_vec[index1]
                simp=Simplex(self.calc_fom, mem, spread*self.simplex_step)
                new_vec, err, iter=simp.minimize(epsilon=self.best_fom/self.simplex_rel_epsilon,
                                                 maxiters=self.simplex_max_iter)
                if self.use_boundaries:
                    # Check so that the parameters lie indside the bounds
                    ok=bitwise_and(self.par_max>new_vec, self.par_min<new_vec)
                    # If not inside make a random re-initilazation of that parameter
                    new_vec=where(ok, new_vec, random.rand(self.n_dim)* \
                                  (self.par_max-self.par_min)+self.par_min)

                new_fom=self.calc_fom(new_vec)
                if new_fom<mem_fom:
                    self.pop_vec[index1]=new_vec.copy()
                    self.fom_vec[index1]=new_fom
                    if new_fom<self.best_fom:
                        self.best_fom=new_fom
                        self.best_vec=new_vec.copy()
                        self.new_best=True

    def simplex_best_1_bin(self, index):
        return self.best_1_bin(index)

    def jade_update_pop(self, index):
        ''' A modified update pop to handle the JADE variation of Differential evoluion'''
        fom=self.trial_fom[index]
        if fom<self.fom_vec[index]:
            self.pop_vec[index]=self.trial_vec[index].copy()
            self.fom_vec[index]=fom
            self.updated_kr.append(self.kr_vec[index])
            self.updated_km.append(self.km_vec[index])
            if fom<self.best_fom:
                self.new_best=True
                self.best_vec=self.trial_vec[index].copy()
                self.best_fom=fom

    def jade_init_new_generation(self, gen):
        ''' A modified generation update for jade'''
        # print 'inits generation: ', gen, self.n_pop
        if gen>1:
            updated_kms=array(self.updated_km)
            updated_krs=array(self.updated_kr)
            if len(updated_kms)!=0:
                self.km=(1.0-self.c)*self.km+self.c*sum(updated_kms**2)/sum(updated_kms)
                self.kr=(1.0-self.c)*self.kr+self.c*mean(updated_krs)
        self.km_vec=abs(self.km+random.standard_cauchy(self.n_pop)*0.1)
        self.kr_vec=self.kr+random.normal(size=self.n_pop)*0.1
        # print self.km_vec, self.kr_vec
        iprint('km: ', self.km, ', kr: ', self.kr)
        # self.km_vec = (self.km_vec >= 1)*1 + (self.km_vec < 1)*self.km_vec
        self.km_vec=where(self.km_vec>0, self.km_vec, 0)
        self.km_vec=where(self.km_vec<1, self.km_vec, 1)
        self.kr_vec=where(self.kr_vec>0, self.kr_vec, 0)
        self.kr_vec=where(self.kr_vec<1, self.kr_vec, 1)

        self.updated_kr=[]
        self.updated_km=[]

    def jade_best(self, index):
        vec=self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1=int(random.rand(1)*self.n_pop)
        index2=int(random.rand(1)*len(self.par_evals))
        # Make sure it is not the same vector 
        # while index2 == index1:
        #    index2 = int(random.rand(1)*self.n_pop)

        # Calculate the mutation vector according to the best/1 scheme
        # print len(self.km_vec), index, len(self.par_evals),  index2
        mut_vec=vec+self.km_vec[index]*(self.best_vec-vec)+self.km_vec[index]*(
                self.pop_vec[index1]-self.par_evals[index2])

        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine=random.rand(self.n_dim)<self.kr_vec[index]
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial=where(recombine, mut_vec, vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok=bitwise_and(self.par_max>trial, self.par_min<trial)
            # If not inside make a random re-initilazation of that parameter
            trial=where(ok, trial, random.rand(self.n_dim)* \
                        (self.par_max-self.par_min)+self.par_min)
        self.trial_vec[index]=trial
        # return trial

    def best_1_bin(self, index):
        '''best_1_bin(self, vec) --> trial [1D array]
        
        The default create_trial function for this class. 
        uses the best1bin method to create a new vector from the population.
        '''
        vec=self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1=int(random.rand(1)*self.n_pop)
        index2=int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2==index1:
            index2=int(random.rand(1)*self.n_pop)

        # Calculate the mutation vector according to the best/1 scheme
        mut_vec=self.best_vec+self.km*(
                self.pop_vec[index1]-self.pop_vec[index2])

        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine=random.rand(self.n_dim)<self.kr
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial=where(recombine, mut_vec, vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok=bitwise_and(self.par_max>trial, self.par_min<trial)
            # If not inside make a random re-initilazation of that parameter
            trial=where(ok, trial, random.rand(self.n_dim)* \
                        (self.par_max-self.par_min)+self.par_min)

        self.trial_vec[index]=trial
        # return trial

    def best_either_or(self, index):
        '''best_either_or(self, vec) --> trial [1D array]
        
        The either/or scheme for creating a trial. Using the best vector
        as base vector.
        '''
        vec=self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1=int(random.rand(1)*self.n_pop)
        index2=int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2==index1:
            index2=int(random.rand(1)*self.n_pop)

        if random.rand(1)<self.pf:
            # Calculate the mutation vector according to the best/1 scheme
            trial=self.best_vec+self.km*(
                    self.pop_vec[index1]-self.pop_vec[index2])
        else:
            # Trying something else out more like normal recombination
            trial=vec+self.kr*(
                    self.pop_vec[index1]+self.pop_vec[index2]-2*vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok=bitwise_and(self.par_max>trial, self.par_min<trial)
            # If not inside make a random re-initilazation of that parameter
            trial=where(ok, trial, random.rand(self.n_dim)* \
                        (self.par_max-self.par_min)+self.par_min)
        self.trial_vec[index]=trial
        # return trial

    def rand_1_bin(self, index):
        '''best_1_bin(self, vec) --> trial [1D array]
        
        The default create_trial function for this class. 
        uses the best1bin method to create a new vector from the population.
        '''
        vec=self.pop_vec[index]
        # Create mutation vector
        # Select three random vectors for the mutation
        index1=int(random.rand(1)*self.n_pop)
        index2=int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2==index1:
            index2=int(random.rand(1)*self.n_pop)
        index3=int(random.rand(1)*self.n_pop)
        while index3==index1 or index3==index2:
            index3=int(random.rand(1)*self.n_pop)

        # Calculate the mutation vector according to the rand/1 scheme
        mut_vec=self.pop_vec[index3]+self.km*(
                self.pop_vec[index1]-self.pop_vec[index2])

        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine=random.rand(self.n_dim)<self.kr
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial=where(recombine, mut_vec, vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok=bitwise_and(self.par_max>trial, self.par_min<trial)
            # If not inside make a random re-initilazation of that parameter
            trial=where(ok, trial, random.rand(self.n_dim)* \
                        (self.par_max-self.par_min)+self.par_min)
        self.trial_vec[index]=trial
        # return trial

    def rand_either_or(self, index):
        '''rand_either_or(self, vec) --> trial [1D array]
        
        random base vector either/or trial scheme
        '''
        vec=self.pop_vec[index]
        # Create mutation vector
        # Select two random vectors for the mutation
        index1=int(random.rand(1)*self.n_pop)
        index2=int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2==index1:
            index2=int(random.rand(1)*self.n_pop)
        index0=int(random.rand(1)*self.n_pop)
        while index0==index1 or index0==index2:
            index0=int(random.rand(1)*self.n_pop)

        if random.rand(1)<self.pf:
            # Calculate the mutation vector according to the best/1 scheme
            trial=self.pop_vec[index0]+self.km*(
                    self.pop_vec[index1]-self.pop_vec[index2])
        else:
            # Calculate a continous recomibination
            # Trying something else out more like normal recombination
            trial=self.pop_vec[index0]+self.kr*(
                    self.pop_vec[index1]+self.pop_vec[index2]-2*self.pop_vec[index0])
            # trial = vec + self.kr*(self.pop_vec[index1]\
            #        + self.pop_vec[index2] - 2*vec)

        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok=bitwise_and(self.par_max>trial, self.par_min<trial)
            # If not inside make a random re-initilazation of that parameter
            trial=where(ok, trial, random.rand(self.n_dim)* \
                        (self.par_max-self.par_min)+self.par_min)
        self.trial_vec[index]=trial
        # return trial

    # Different function for acessing and setting parameters that
    # the user should have control over.

    def set_text_output_func(self, func):
        '''set_text_output_func(self, func) --> None
        
        Set the output function for the text output from the optimizer.
        Should be a function that takes a string as input argument.
        The default function is a simple print statement.
        '''
        self.text_output=func

    def set_plot_output_func(self, func):
        '''set_plot_output_func(self, func) --> None

        Set the output function for the plot output from the optimizer.
        Should take the an instance of solver as input.
        The default function is no output whatsoever
        '''
        self.plot_output=func

    def set_parameter_output_func(self, func):
        '''set_parameters_output_func(self, func) --> None

        Set the output function for the parameters output from the optimizer.
        Should take the an instance of solver as input.
        The default function is no output whatsoever
        '''
        self.parameter_output=func

    def set_fitting_ended_func(self, func):
        '''set_fitting_ended_func(self, func) --> None

        Set the function when the optimizer has finsihed the fitting.
        Should take the an instance of solver as input.
        The default function is no output whatsoever
        '''
        self.fitting_ended=func

    def set_autosave_func(self, func):
        '''set_autosave_func(self, func) --> None
        
        Set the function that the optimizer uses to do an autosave
        of the current fit. Function func should not take any arguments.
        '''
        self.autosave=func

    # Some get functions

    def get_model(self):
        '''get_model(self) --> model
        Getter that returns the model in use in solver.
        '''
        return self.model

    def get_fom_log(self):
        '''get_fom_log(self) -->  fom [array]
        Returns the fom as a fcn of iteration in an array. 
        Last element last fom value
        '''
        return array(self.fom_log)

    def get_create_trial(self, index=False):
        '''get_create_trial(self, index = False) --> string or int
        
        returns the current create trial function name if index is False as
        a string or as index in the mutation_schemes list.
        '''
        pos=self.mutation_schemes.index(self.create_trial)
        if index:
            # return the position
            return pos
        else:
            # return the name
            return self.mutation_schemes[pos].__name__

    def set_km(self, val):
        '''set_km(self, val) --> None
        '''
        self.km=val

    def set_kr(self, val):
        '''set_kr(self, val) --> None
        '''
        self.kr=val

    def set_create_trial(self, val):
        '''set_create_trial(self, val) --> None
        
        Raises LookupError if the value val [string] does not correspond
        to a mutation scheme/trial function
        '''
        # Get the names of the available functions
        names=[f.__name__ for f in self.mutation_schemes]
        # Find the postion of val

        pos=names.index(val)
        self.create_trial=self.mutation_schemes[pos]
        if val=='jade_best':
            self.update_pop=self.jade_update_pop
            self.init_new_generation=self.jade_init_new_generation
        elif val=='simplex_best_1_bin':
            self.init_new_generation=self.simplex_init_new_generation
            self.update_pop=self.standard_update_pop
        else:
            self.init_new_generation=self.standard_init_new_generation
            self.update_pop=self.standard_update_pop

    def set_pop_mult(self, val):
        '''set_pop_mult(self, val) --> None
        '''
        self.pop_mult=val

    def set_pop_size(self, val):
        '''set_pop_size(self, val) --> None
        '''
        self.pop_size=int(val)

    def set_max_generations(self, val):
        '''set_max_generations(self, val) --> None
        '''
        self.max_generations=int(val)

    def set_max_generation_mult(self, val):
        '''set_max_generation_mult(self, val) --> None
        '''
        self.max_generation_mult=val

    def set_sleep_time(self, val):
        '''set_sleep_time(self, val) --> None
        '''
        self.sleep_time=val

    def set_max_log(self, val):
        '''Sets the maximum number of logged elements
        '''
        self.max_log=val

    def set_use_pop_mult(self, val):
        '''set_use_pop_mult(self, val) --> None
        '''
        self.use_pop_mult=val

    def set_use_max_generations(self, val):
        '''set_use_max_generations(self, val) --> None
        '''
        self.use_max_generations=val

    def set_use_start_guess(self, val):
        '''set_use_start_guess(self, val) --> None
        '''
        self.use_start_guess=val

    def set_use_boundaries(self, val):
        '''set_use_boundaries(self, val) --> None
        '''
        self.use_boundaries=val

    def set_use_autosave(self, val):
        '''set_use_autosave(self, val) --> None
        '''
        self.use_autosave=val

    def set_autosave_interval(self, val):
        '''set_autosave_interval(self, val) --> None
        '''
        self.autosave_interval=int(val)

    def set_use_parallel_processing(self, val):
        '''set_use_parallel_processing(self, val) --> None
        '''
        if __parallel_loaded__:
            self.use_parallel_processing=val
            self.use_mpi=False if val else self.use_mpi
        else:
            self.use_parallel_processing=False

    def set_use_mpi(self, val):
        """Sets if mpi should use for parallel optimization"""
        if __mpi_loaded__:
            self.use_mpi=val
            self.use_parallel_processing=False if val else self.use_parallel_processing
        else:
            self.use_mpi=False

    def set_processes(self, val):
        '''set_processes(self, val) --> None
        '''
        self.processes=int(val)

    def set_chunksize(self, val):
        '''set_chunksize(self, val) --> None
        '''
        self.chunksize=int(val)

    def set_fom_allowed_dis(self, val):
        '''set_chunksize(self, val) --> None
        '''
        self.fom_allowed_dis=float(val)

    def __repr__(self):
        output="Differential Evolution Optimizer:\n"
        for gname, group in self.parameter_groups:
            output+='    %s:\n'%gname
            for attr in group:
                output+='        %-30s %s\n'%(attr, getattr(self, attr))
        return output

    @property
    def widget(self):
        return self._repr_ipyw_()

    def _repr_ipyw_(self):
        import ipywidgets as ipw
        entries=[]
        for gname, group in self.parameter_groups:
            gentries=[ipw.HTML("<b>%s:</b>"%gname)]
            for attr in group:
                val=eval('self.%s'%attr, globals(), locals())
                if type(val) is bool:
                    item=ipw.Checkbox(value=val, indent=False, description=attr, layout=ipw.Layout(width='24ex'))
                    entry=item
                elif type(val) is int:
                    entry=ipw.IntText(value=val, layout=ipw.Layout(width='18ex'))
                    item=ipw.VBox([ipw.Label(attr), entry])
                elif type(val) is float:
                    entry=ipw.FloatText(value=val, layout=ipw.Layout(width='18ex'))
                    item=ipw.VBox([ipw.Label(attr), entry])
                elif attr=='method':
                    entry=ipw.Dropdown(value=val, options=self.methods, layout=ipw.Layout(width='18ex'))
                    item=ipw.VBox([ipw.Label(attr), entry])
                else:
                    entry=ipw.Text(value=val, layout=ipw.Layout(width='14ex'))
                    item=ipw.VBox([ipw.Label(attr), entry])
                entry.change_item=attr
                entry.observe(self._ipyw_change, names='value')
                gentries.append(item)
            entries.append(ipw.VBox(gentries, layout=ipw.Layout(width='26ex')))
        return ipw.VBox([ipw.HTML("<h3>Optimizer Settings:</h3>"), ipw.HBox(entries)])

    def _ipyw_change(self, change):
        exec('self.%s=change.new'%change.owner.change_item)

# ==============================================================================
# Functions that is needed for parallel processing!
def parallel_init(model_copy, use_cuda):
    '''parallel_init(model_copy) --> None
    
    parallel initilization of a pool of processes. The function takes a
    pickle safe copy of the model and resets the script module and the compiles
    the script and creates function to set the variables.
    '''
    if use_cuda:
        # activate cuda in subprocesses
        from genx.models.lib import paratt_cuda
        from genx.models.lib import neutron_cuda
        from models.lib import paratt, neutron_refl
        paratt.Refl=paratt_cuda.Refl
        paratt.ReflQ=paratt_cuda.ReflQ
        paratt.Refl_nvary2=paratt_cuda.Refl_nvary2
        neutron_refl.Refl=neutron_cuda.Refl
        from genx.models.lib import paratt, neutron_refl
        paratt.Refl=paratt_cuda.Refl
        paratt.ReflQ=paratt_cuda.ReflQ
        paratt.Refl_nvary2=paratt_cuda.Refl_nvary2
        neutron_refl.Refl=neutron_cuda.Refl
    global model, par_funcs
    model=model_copy
    model._reset_module()
    model.simulate()
    (par_funcs, start_guess, par_min, par_max)=model.get_fit_pars()
    # print 'Sucess!'

def parallel_calc_fom(vec):
    '''parallel_calc_fom(vec) --> fom (float)
    
    function that is used to calculate the fom in a parallel process.
    It is a copy of calc_fom in the DiffEv class
    '''
    global model, par_funcs
    # print 'Trying to set parameters'
    # set the parameter values in the model
    list(map(lambda func, value: func(value), par_funcs, vec))
    # print 'Trying to evaluate'
    # evaluate the model and calculate the fom
    fom=model.evaluate_fit_func()

    return fom

# ==============================================================================
def default_text_output(text):
    iprint(text)
    sys.stdout.flush()

def default_plot_output(solver):
    pass

def default_parameter_output(solver):
    pass

def default_fitting_ended(solver):
    pass

def defualt_autosave():
    pass

def _calc_fom(model, vec, par_funcs):
    '''
    Function to calcuate the figure of merit for parameter vector
    vec.
    '''
    # Set the paraemter values
    list(map(lambda func, value: func(value), par_funcs, vec))

    return model.evaluate_fit_func()

# ==============================================================================
# BEGIN: class CircBuffer
class CircBuffer:
    '''A buffer with a fixed length to store the logging data from the diffev 
    class. Initilized to a maximumlength after which it starts to overwrite
    the data again.
    '''

    def __init__(self, maxlen, buffer=None):
        '''Inits the class with a certain maximum length maxlen.
        '''
        self.maxlen=int(maxlen)
        self.pos=-1
        self.filled=False
        if buffer is None:
            self.buffer=zeros((self.maxlen,))
        else:
            if len(buffer)!=0:
                self.buffer=array(buffer).repeat(
                    ceil(self.maxlen/(len(buffer)*1.0)), 0)[:self.maxlen]
                self.pos=len(buffer)-1
            else:
                self.buffer=zeros((self.maxlen,)+buffer.shape[1:])

    def reset(self, buffer=None):
        '''Resets the buffer to the initial state
        '''
        self.pos=-1
        self.filled=False
        # self.buffer = buffer
        if buffer is None:
            self.buffer=zeros((self.maxlen,))
        else:
            if len(buffer)!=0:
                self.buffer=array(buffer).repeat(
                    ceil(self.maxlen/(len(buffer)*1.0)), 0)[:self.maxlen]
                self.pos=len(buffer)-1
            else:
                self.buffer=zeros((self.maxlen,)+buffer.shape[1:])

    def append(self, item, axis=None):
        '''Appends an element to the last position of the buffer
        '''
        new_pos=(self.pos+1)%self.maxlen
        if len(self.buffer)>=self.maxlen:
            if self.pos>=(self.maxlen-1):
                self.filled=True
            self.buffer[new_pos]=array(item).real
        else:
            self.buffer=append(self.buffer, item, axis=axis)
        self.pos=new_pos

    def array(self):
        '''returns an ordered array instead of the circular
        working version
        '''
        if self.filled:
            return r_[self.buffer[self.pos+1:], self.buffer[:self.pos+1]]
        else:
            return r_[self.buffer[:self.pos+1]]

    def copy_from(self, object):
        '''Add copy support
        '''
        if type(object)==type(array([])):
            self.buffer=object[-self.maxlen:]
            self.pos=len(self.buffer)-1
            self.filled=self.pos>=(self.maxlen-1)
        elif object.__class__==self.__class__:
            # Check if the buffer has been removed.
            if len(object.buffer)==0:
                self.__init__(object.maxlen, object.buffer)
            else:
                self.buffer=object.buffer.copy()
                self.maxlen=object.maxlen
                self.pos=object.pos
                try:
                    self.filled=object.filled
                except:
                    self.filled=False
        else:
            raise TypeError('CircBuffer support only copying from CircBuffer'
                            ' and arrays.')

    def __len__(self):
        if self.filled:
            return len(self.buffer)
        else:
            return (self.pos>0)*self.pos

    def __getitem__(self, key):
        return self.array().__getitem__(key)

# END: class CircBuffer
# ==============================================================================
class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass

class ErrorBarError(GenericError):
    '''Error class for the fom evaluation'''

    def __init__(self):
        ''' __init__(self) --> None'''
        # self.error_message = error_message

    def __str__(self):
        text='Could not evaluate the error bars. A fit has to be made '+ \
             'before they can be calculated'
        return text

if __name__=='__main__':
    import h5py

    d=DiffEv()
    iprint(arange(10))
    d.fom_evals.copy_from(arange(10))
    iprint(d.fom_evals.buffer, d.fom_evals.pos, d.fom_evals.filled)
    d.km=10
    iprint(d.fom_evals.array())
    f=h5py.File('myfile.hdf5', 'w')
    dic=f.create_group('optimizer')
    d.write_h5group(dic)
    f.close()

    d=DiffEv()
    f=h5py.File('myfile.hdf5', 'r')
    dic=f['optimizer']
    d.read_h5group(dic)
    iprint(d.km)
    iprint(d.fom_evals.array(), d.fom_evals.pos, d.fom_evals.filled)
    f.close()
