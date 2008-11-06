'''File: diffev.py an implementation of the differential evolution algoithm
for fitting.
Programmed by: Matts Bjorck
Last changed: 2008 06 01
'''

from numpy import *
import thread
import time

# Parallel python
#import pp

#==============================================================================
# class: DiffEv
class DiffEv:
    '''
    Class DiffEv
    Contains the implemenetation of the differential evolution algorithm.
    It also contains thread support which is activated by the start_fit 
    function.
    '''
    def __init__(self):
        # Mutation schemes implemented
        self.mutation_schemes = [self.best_1_bin, self.rand_1_bin,\
            self.best_either_or, self.rand_either_or]
        
        self.km = 0.7 # Mutation constant
        self.kr = 0.7 # Cross over constant
        self.pf = 0.5 # probablility for mutation
        
        # Flag to choose beween the two alternatives below
        self.use_pop_mult = False
        self.pop_mult = 3 # Set the pop_size to pop_mult * # free parameters
        self.pop_size = 10 # Set the pop_size only
        
        # Flag to choose between the two alternatives below
        self.use_max_generations = False 
        self.max_generations = 500 # Use a fixed # of iterations
        self.max_generation_mult = 6 # A mult const for max number of iter
        
        # Flag to choose whether or not to use a starting guess
        self.use_start_guess = True
        # Flag to choose wheter or not to use the boundaries
        self.use_boundaries = True
        
        # Sleeping time for every generation
        self.sleep_time = 0.2
        
        # Functions that are user definable
        self.plot_output = default_plot_output
        self.text_output = default_text_output
        self.parameter_output = default_parameter_output
        self.fitting_ended = default_fitting_ended
        
        
        # Definition for the create_trial function
        self.create_trial = self.best_1_bin
        
        # Control flags:
        self.running = False # true if optimization is running
        self.stop = False # true if the optimization should stop
        self.setup_ok = False # True if the optimization have been setup
        
        # Logging variables
        self.fom_log = []
        self.par_evals = array([[]])[0:0]
        self.fom_evals = array([])

    def reset(self):
        ''' reset(self) --> None
        
        Resets the optimizer
        '''
        self.setup_ok = False
        
    def init_fitting(self, model):
        '''
        Function to run before a new fit is started with start_fit.
        It initilaize the population and sets the limits on the number
        of generation and the population size.
        '''
        # Retrive parameters from the model
        (par_funcs, start_guess, par_min, par_max) = model.get_fit_pars()
        
        # Control parameter setup
        self.par_min = array(par_min)
        self.par_max = array(par_max)
        self.par_funcs = par_funcs
        self.model = model
        self.n_dim = len(par_funcs)
        if self.use_pop_mult:
            self.n_pop = int(self.pop_mult*self.n_dim)
        else:
            self.n_pop = int(self.pop_size)
        if self.use_max_generations:
            self.max_gen = int(self.max_generations)
        else:
            self.max_gen = int(self.max_generation_mult*self.n_dim*self.n_pop)
        
        # Starting values setup
        self.pop_vec = [self.par_min + random.rand(self.n_dim)*(self.par_max -\
         self.par_min) for i in range(self.n_pop)]
        
        if self.use_start_guess:
            self.pop_vec[0] = array(start_guess)
            
        self.trial_vec = [zeros(self.n_dim) for i in range(self.n_pop)]
        self.best_vec = self.pop_vec[0]
        
        self.fom_vec = zeros(self.n_dim)
        self.best_fom = 1e20

        # Logging varaibles
        self.fom_log = []
        self.par_evals = array([par_min])[0:0]
        self.fom_evals = array([])
        
        self.text_output('DE initilized')
        
        # Remeber that everything has been setup ok
        self.setup_ok = True
        
        
    def start_fit(self, model):
        '''
        Starts fitting in a seperate thred.
        '''
        # If it is not already running
        if not self.running:
            #Initilize the parameters to fit
            self.init_fitting(model)
            self.stop = False
            # Start fitting in a new thread
            thread.start_new_thread(self.optimize, ())
            # For debugging
            #self.optimize()
            #self.text_output('Starting the fit...')
            #self.running = True
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
            self.stop = True
            self.text_output('Trying to stop the fit...')
        else:
            self.text_output('The fit is not running')
        
    def resume_fit(self):
        '''
        Resumes the fitting if has been stopped with stop_fit.
        '''
        if not self.running:
            self.stop = False
            thread.start_new_thread(self.optimize, ())
            self.text_output('Restarting the fit...')
            self.running = True
            return True
        else:
            self.text_output('Fit is already running, stop and then start')
            return False
        
    def optimize(self):
        '''
        Method implementing the main loop of the differential evolution
        algorithm. Note that this method does not run in a separate thread.
        For threading use start_fit, stop_fit and resume_fit instead.
        '''
        self.text_output('Calculating start FOM ...')
        
        self.running = True
        #print self.pop_vec
        self.fom_vec = [self.calc_fom(vec) for vec in self.pop_vec]
        #print self.fom_vec
        # test for parallel python which will hopefully work some day
        #self.setup_pp()
        #self.calc_pp_fom()
        #print "Sucess?"
        # End test part
        #print self.fom_vec
        best_index = argmin(self.fom_vec)
        #print best_index
        self.best_vec = copy(self.pop_vec[best_index])
        #print self.best_vec
        self.best_fom = self.fom_vec[best_index]
        #print self.best_fom
        self.fom_log= array([[len(self.fom_log),self.best_fom]])
        # Flag to keep track if there has been any improvemnts
        # in the fit - used for updates
        self.new_best = True
        
        self.text_output('Going into optimization ...')
        
        gen = 0 # Just making gen live in this scope as well...
        for gen in range(self.max_gen):
            if self.stop:
                break
            
            t_start = time.clock()
            
            # Create the vectors who will be compared to the 
            # population vectors
            self.trial_vec = [self.create_trial(vec) for vec in self.pop_vec]
            # Calculate the fom of the trial vectors and update the population
            [self.update_pop(index) for index in range(self.n_pop)]
            
            # Add the best value to the fom log
            self.fom_log = r_[self.fom_log,\
                                [[len(self.fom_log),self.best_fom]]]
            
            # Let the model calculate the simulation of the best.
            self.calc_sim(self.best_vec)
            
            # Update the plot data for any gui or other output
            self.plot_output(self)
            self.parameter_output(self)
            
            # Let the optimization sleep for a while
            time.sleep(self.sleep_time)
            
            # Time measurent to track the speed
            t = time.clock() - t_start
            if t > 0:
                speed = self.n_pop/t
            else:
                speed = 999999
            self.text_output('FOM: %.3f Generation: %d Speed: %.1f'%\
            (self.best_fom,gen,speed))
            
            self.new_best = False

        self.text_output('Stopped at Generation: %d ...'%gen)
        
        # Now the optimization has stopped
        self.running = False
        
        # Run application specific clean-up actions
        self.fitting_ended(self)
        
            
    def calc_fom(self, vec):
        '''
        Function to calcuate the figure of merit for parameter vector 
        vec.
        '''
        # Set the parameter values
        map(lambda func, value:func(value), self.par_funcs, vec)
        fom = self.model.evaluate_fit_func()
        
        # Add the evaluation to the logging
        self.par_evals = append(self.par_evals, [vec], axis = 0)
        self.fom_evals = append(self.fom_evals, fom)
        
        return fom 
    
    
    def calc_sim(self, vec):
        ''' calc_sim(self, vec) --> None
        Function that will evaluate the the data points for
        parameters in vec.
        '''
        # Set the paraemter values
        map(lambda func, value:func(value), self.par_funcs, vec)
        
        self.model.evaluate_sim_func()
        
    def setup_pp(self):
        self.job_server = pp.Server()
        print "Starting pp with", self.job_server.get_ncpus(), "workers"
        # Thats it now pp is set up to go ...
    
    def calc_pp_fom(self):
        '''
        Function to calculate the fom in parallel using parallel python
        '''
        jobs = [self.job_server.submit(_calc_fom,\
            (self.model, vec, self.par_funcs)) for vec in self.pop_vec]
        for job in jobs:
            print job()
    
    def calc_error_bar(self, index, fom_level):
        '''calc_error_bar(self, parameter) --> (error_bar_low, error_bar_high)
        
        Calculates the errorbar for one parameter number index. 
        returns a float tuple with the error bars. fom_level is the 
        level which is the upperboundary of the fom is allowed for the
        calculated error.
        '''
        #print self.par_evals.shape, self.par_evals
        #print self.fom_evals.shape, self.fom_evals
        if self.setup_ok: #and len(self.par_evals) != 0:
            par_values = self.par_evals[:,index]        
            #print (self.fom_evals < fom_level).sum()
            values_under_level = compress(self.fom_evals <\
                                    fom_level*self.best_fom, par_values)
            #print values_under_level
            error_bar_low = values_under_level.min() - self.best_vec[index]
            error_bar_high = values_under_level.max() - self.best_vec[index]
            return (error_bar_low, error_bar_high)
        else:
            raise ErrorBarsError()
        
        
    def update_pop(self, index):
        '''
        Function to update population vector index. calcs the figure of merit
        and compares it to the current population vector and also cehecks
        if it is better than the current best.
        '''
        fom = self.calc_fom(self.trial_vec[index])
        if fom < self.fom_vec[index]:
            self.pop_vec[index] = self.trial_vec[index].copy()
            self.fom_vec[index] = fom
            if fom < self.best_fom:
                self.new_best = True
                self.best_vec = self.trial_vec[index].copy()
                self.best_fom = fom

    
    def best_1_bin(self, vec):
        '''best_1_bin(self, vec) --> trial [1D array]
        
        The default create_trial function for this class. 
        uses the best1bin method to create a new vector from the population.
        '''
        # Create mutation vector
        # Select two random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)
            
        # Calculate the mutation vector according to the best/1 scheme
        mut_vec = self.best_vec + self.km*(self.pop_vec[index1]\
         - self.pop_vec[index2])
        
        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine = random.rand(self.n_dim)<self.kr
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial = where(recombine, mut_vec, vec)
        
        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        
        return trial
    
    def best_either_or(self, vec):
        '''best_either_or(self, vec) --> trial [1D array]
        
        The either/or scheme for creating a trial. Using the best vector
        as base vector.
        '''
        # Create mutation vector
        # Select two random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)
        
        if random.rand(1) < self.pf:
            # Calculate the mutation vector according to the best/1 scheme
            trial = self.best_vec + self.km*(self.pop_vec[index1]\
            - self.pop_vec[index2])
        else:
            # Trying something else out more like normal recombination
            trial = vec + self.kr*(self.pop_vec[index1]\
            + self.pop_vec[index2] - 2*vec)
        
        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        
        return trial
        
    def rand_1_bin(self, vec):
        '''best_1_bin(self, vec) --> trial [1D array]
        
        The default create_trial function for this class. 
        uses the best1bin method to create a new vector from the population.
        '''
        # Create mutation vector
        # Select three random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)
        index3 = int(random.rand(1)*self.n_pop)
        while index3 == index1 or index3 == index2:
            index3 = int(random.rand(1)*self.n_pop)
            
        # Calculate the mutation vector according to the rand/1 scheme
        mut_vec = self.pop_vec[index3] + self.km*(self.pop_vec[index1]\
         - self.pop_vec[index2])
        
        # Binomial test to detemine which parameters to change
        # given by the recombination constant kr
        recombine = random.rand(self.n_dim)<self.kr
        # Make sure at least one parameter is changed
        recombine[int(random.rand(1)*self.n_dim)]=1
        # Make the recombination
        trial = where(recombine, mut_vec, vec)
        
        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        
        return trial
    
    def rand_either_or(self, vec):
        '''rand_either_or(self, vec) --> trial [1D array]
        
        random base vector either/or trial scheme
        '''
        # Create mutation vector
        # Select two random vectors for the mutation
        index1 = int(random.rand(1)*self.n_pop)
        index2 = int(random.rand(1)*self.n_pop)
        # Make sure it is not the same vector 
        while index2 == index1:
            index2 = int(random.rand(1)*self.n_pop)
        index0 = int(random.rand(1)*self.n_pop)
        while index0 == index1 or index0 == index2:
            index0 = int(random.rand(1)*self.n_pop)
        
        if random.rand(1) < self.pf:
            # Calculate the mutation vector according to the best/1 scheme
            trial = self.pop_vec[index0] + self.km*(self.pop_vec[index1]\
            - self.pop_vec[index2])
        else:
            # Calculate a continous recomibination
            # Trying something else out more like normal recombination
            #trial = self.pop_vec[index0] + self.kr*(self.pop_vec[index1]\
            #+ self.pop_vec[index2] - 2*self.pop_vec[index0])
            trial = vec + self.kr*(self.pop_vec[index1]\
                    + self.pop_vec[index2] - 2*vec)
        
        # Implementation of constrained optimization
        if self.use_boundaries:
            # Check so that the parameters lie indside the bounds
            ok = bitwise_and(self.par_max > trial, self.par_min < trial)
            # If not inside make a random re-initilazation of that parameter
            trial = where(ok, trial, random.rand(self.n_dim)*\
            (self.par_max - self.par_min) + self.par_min)
        
        return trial
    
    
    # Different function for acessing and setting parameters that 
    # the user should have control over.
        
    def set_text_output_func(self, func):
        '''set_text_output_func(self, func) --> None
        
        Set the output function for the text output from the optimizer.
        Should be a function that takes a string as input argument.
        The default function is a simple print statement.
        '''
        self.text_output = func
            
    def set_plot_output_func(self, func):
       '''set_plot_output_func(self, func) --> None
    
       Set the output function for the plot output from the optimizer.
       Should take the an instance of solver as input.
       The default function is no output whatsoever
       '''
       self.plot_output = func
         
    
    def set_parameter_output_func(self, func):
       '''set_parameters_output_func(self, func) --> None
    
       Set the output function for the parameters output from the optimizer.
       Should take the an instance of solver as input.
       The default function is no output whatsoever
       '''
       self.parameter_output = func
         
    
    def set_fitting_ended_func(self, func):
       '''set_fitting_ended_func(self, func) --> None
    
       Set the function when the optimizer has finsihed the fitting.
       Should take the an instance of solver as input.
       The default function is no output whatsoever
       '''
       self.fitting_ended = func
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
    
    def get_create_trial(self, index = False):
        '''get_create_trial(self, index = False) --> string or int
        
        returns the current create trial function name if index is False as
        a string or as index in the mutation_schemes list.
        '''
        pos = self.mutation_schemes.index(self.create_trial)
        if index:
            # return the position
            return pos
        else:
            # return the name
            return self.mutation_schemes[pos].__name__
    
    def set_km(self, val):
        '''set_km(self, val) --> None
        '''
        self.km = val
    
    def set_kr(self, val):
        '''set_kr(self, val) --> None
        '''
        self.kr = val
        
    def set_create_trial(self, val):
        '''set_create_trial(self, val) --> None
        
        Raises LookupError if the value val [string] does not correspond
        to a mutation scheme/trial function
        '''
        # Get the names of the available functions
        names = [f.__name__ for f in self.mutation_schemes]
        # Find the postion of val
        pos = names.index(val)
        self.create_trial = self.mutation_schemes[pos]
        
    def set_pop_mult(self, val):
        '''set_pop_mult(self, val) --> None
        '''
        self.pop_mult = val
    
    def set_pop_size(self, val):
        '''set_pop_size(self, val) --> None
        '''
        self.pop_size = int(val)
        
    def set_max_generations(self, val):
        '''set_max_generations(self, val) --> None
        '''
        self.max_generations = int(val)
        
    def set_max_generation_mult(self, val):
        '''set_max_generation_mult(self, val) --> None
        '''
        self.max_generation_mult = val
        
    def set_sleep_time(self, val):
        '''set_sleep_time(self, val) --> None
        '''
        self.sleep_time = val
        
    def set_use_pop_mult(self, val):
        '''set_use_pop_mult(self, val) --> None
        '''
        self.use_pop_mult = val
        
    def set_use_max_generations(self, val):
        '''set_use_max_generations(self, val) --> None
        '''
        self.use_max_generations = val
        
    def set_use_start_guess(self, val):
        '''set_use_start_guess(self, val) --> None
        '''
        self.use_start_guess = val
    
    def set_use_boundaries(self, val):
        '''set_use_boundaries(self, val) --> None
        '''
        self.use_boundaries = val
        
    
    
#==============================================================================
def default_text_output(text):
    print text

def default_plot_output(solver):
    pass
    
def default_parameter_output(solver):
    pass
    
def default_fitting_ended(solver):
    pass
    

def _calc_fom(model, vec, par_funcs):
        '''
        Function to calcuate the figure of merit for parameter vector 
        vec.
        '''
        # Set the paraemter values
        map(lambda func, value:func(value), par_funcs, vec)
        
        return model.evaluate_fit_func()

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
        #self.error_message = error_message
    
    def __str__(self):
        text = 'Could not evaluate the error bars. A fit has to be made' +\
                'before they can be calculated'
        return text