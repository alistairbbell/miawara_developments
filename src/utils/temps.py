import datetime as dt
import numpy as np
from datetime import tzinfo

def datetimeIterator(from_date=None, to_date=None, delta=dt.timedelta(hours=1)):
    while from_date <= to_date:
        yield from_date
        from_date = from_date + delta
    return

def calculate_R2(q):
    Rm = (1 + (0.61*  q)) * R_dry
    return Rm
    
def between_heights(array, append_end = True, append_val = None):
    if append_val == None:
        if append_end == True:
            append_val = array[-1]
        else:
            append_val = array[0]
    if append_end == True:
        array = np.append(array, append_val)
    else:
        array = np.append(append_val, array)
    return_array = [np.mean((array[i], array[i+1])) for i in range(len(array)-1)]
    return return_array


def between_heights2(array, append_end = 0, append_beg = 9999):
    array = np.append(append_beg, array)
    array = np.append(array, append_end)
    return_array = [np.mean((array[i], array[i+1])) for i in range(len(array)-1)]
    return return_array

def nearest(items, pivot, minimum = 2000):
    '''Returns index of items closest to pivot values. items and pivot must be
    specified lists.'''
    mylist = np.ones(len(pivot)) * -1
    for index, i in enumerate(pivot, 0):
        #print(i)
        try:
            temp = min(items, key=lambda x: abs(x - i))
            if abs(temp - i) < minimum:
                mylist[index] = int(items.index((temp)))
            else:
                mylist[index] = np.nan #(int(-1))
                #print('outside minimum')
        except Exception:
            mylist[index] = np.nan
            print('exception raised')
    mylist = list(mylist)  
    for i in range(len(mylist)):
      try: mylist[i] = int(mylist[i])
      except: 
        pass
    return list(mylist)

def window(items, pivot, window = dt.timedelta(0, 3*3600)):
    '''returns index of items within pivot window'''
    items = list(items)
    mylist = []
    for i in items:
        if abs(i - pivot) < window:
            mylist.append(items.index(i))
    return mylist
    
def time_mod(time, delta, epoch=None):
    if epoch is None:
        epoch = dt.datetime(1970, 1, 1)
    return (time - epoch).seconds % delta.seconds

def time_round(time, delta, epoch=None):
    time = time.replace(microsecond = 0)
    mod = time_mod(time, delta, epoch)
    if mod < (delta.seconds / 2):
       return time - dt.timedelta(seconds = mod)
    return (time + dt.timedelta(seconds = (delta.seconds - mod)))

def average_between_df(interpolation_heights, dataframe, variable, minimum = 40):
  ret_array = np.ones(len(interpolation_heights))
  my_bet_heights = between_heights2(interpolation_heights, append_beg = interpolation_heights[0])
  for i in range(len(interpolation_heights)):
    ret_array[i] = dataframe[(dataframe.Height<my_bet_heights[i]) & (dataframe.Height>my_bet_heights[i+1])][variable].mean()
  return ret_array

def index_else_nan(array, indices, axis = 1, filler = np.nan, specify_height = False):
  mylist = 0
  if axis == None:
    filler = np.nan
    start_array = np.zeros(0)
    print(start_array.shape)
    for i in indices:
      if ~np.isnan(i):
        start_array = np.append(start_array, array[ int(i):int(i+1)])
      else:
        start_array = np.append(start_array, filler)
  if axis == 1:
    filler = np.ones((len(array[:,1]),1))   *np.nan
    start_array = np.zeros((len(array[:,0]), 0))
    print(start_array.shape)
    for i in indices:
      if ~np.isnan(i):
        print( array[:, int(i):int(i+1)].shape)
        start_array = np.append(start_array, array[:, int(i):int(i+1)], axis = 1)
        print(start_array.shape)
      else:
        start_array = np.append(start_array, filler, axis = 1)
  elif axis == 0:
    filler = np.ones((1,len(array[1,:]))) * np.nan
    start_array = np.zeros((0, (len(array[0,:]))))
    for i in indices:
      if ~np.nan(i):
        start_array = np.append(start_array, array[int(i):int(i+1),:], axis = 0)
      else:
        start_array = np.append(start_array, filler, axis = 0)
  if axis == 'both':
    filler = np.nan
    start_array = np.zeros(0)
    #print(start_array.shape)
    for i in range(len(indices[0])):
      #print((indices[0][i]),(indices[1][i]))
      if ((~np.isnan(indices[0][i])) & (~np.isnan(indices[1][i]))):
        mylist +=1
        start_array = np.append(start_array, array[ int(indices[0][i]),int(indices[1][i])])
        if ((array[ int(indices[0][i]),int(indices[1][i])] >= 0) & (array[ int(indices[0][i]),int(indices[1][i])] <= 10)):
          print(i)
      else:
        #print('nan encountered')

        start_array = np.append(start_array, filler)
  print( mylist)
  return start_array

def myround(x, base=5):
    return base * round(x/base)
  
def specifc_to_abs(P,T,q,clw,cli,rain,snow,graupel):
  R = calculate_R2(q)
  den = P/(R*T)
  return clw*den


def abs_to_specific(P,T,q,clw,cli,rain,snow,graupel):
  R = calculate_R2(q, clw, cli, rain,snow,graupel)
  den = P/(R*T)
  return clw/den

def air_den(P,T,q,LWC):
  R = calculate_R2(q, LWC,0,0,0,0)
  return P/(R*T)
