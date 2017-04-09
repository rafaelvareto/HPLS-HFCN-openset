from multiprocessing.pool import ThreadPool

def foo(word, number):
    print word*number
    return number

def starfoo(args):
    """ 

    We need this because map only supports calling functions with one arg. 
    We need to pass two args, so we use this little wrapper function to
    expand a zipped list of all our arguments.

    """    
    return foo(*args)

words = ['hello', 'world', 'test', 'word', 'another test']
numbers = [1,2,3,4,5]
pool = ThreadPool(5)
# We need to zip together the two lists because map only supports calling functions
# with one argument. In Python 3.3+, you can use starmap instead.
results = pool.map(starfoo, zip(words, numbers))
print results

pool.close()
pool.join()