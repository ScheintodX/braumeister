from datetime import datetime, timedelta
from functools import partial, wraps

import argparse
import pandas as pd
import loader
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution

from pandas import DataFrame
import math

K = 273

def _K( c ):
    return c + K

def _C( k ):
    return k - K

T_room:float = _K( 20 )
Dt:float = 30 #s intervals


class Kiln:

    c:float = 920
    m:float = 100
    Q:float = 18000

    E:float = T_room * c * m #Ws


    def tick( self, dt:float, powerfactor:float, damper:float, A1:float, A2:float, A3:float, A4:float ):

        dE:float = dt * powerfactor * self.Q
        dE -= dt * A1 * powerfactor
        dE -= dt * A2 * math.pow(self.T() - T_room, 1)
        dE -= dt * A3 * math.pow(self.T() - T_room, 2)
        dE -= dt * A4 * math.pow(self.T() - T_room, 3)

        self.E += dE

    def T( self ):
        r = self.E / self.c / self.m
        if( r > 99999999 ): return 99999999
        if( r < -99999999 ): return -99999999
        return r

    def C( self ):
        return _C( self.T() )

    def setT( self, k:float ):
        self.E = k * self.c * self.m

    def setC( self, c:float ):
        self.setT( _K( c ) )


def timerange( start:str, end:str, step:int ):

    start_time = datetime.strptime(start, "%Y-%m-%d %H:%M:%S") if type( start ) is str else start
    end_time = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    step_delta = timedelta(seconds=step)

    current_time = start_time
    while current_time <= end_time:
        yield current_time.strftime("%Y-%m-%d %H:%M:%S")
        current_time += step_delta


def run_kiln( kiln:Kiln, ref:DataFrame, A1:float, A2:float, A3:float, A4:float):

    data:dict = []

    for time, row in ref.iterrows():

        powerfactor = row['powerfactor']
        damper = row['damper']
        T = row['temperature']

        kiln.tick( Dt, powerfactor, damper, A1, A2, A3, A4 )

        data.append( {
                "time": time,
                "sim": kiln.C(),
                "error": math.fabs( T - kiln.C())
        } )

    df:DataFrame = DataFrame( data )
    df.set_index( "time", inplace=True )

    return df


count:int = 0

def run_and_concat( ref:DataFrame, A1:float, A2:float, A3:float, A4:float):

    global count

    count += 1

    print( A1, A2, A3, A4 )

    kiln:Kiln = Kiln()
    kiln.setC( ref.iloc[0]['temperature'] )

    ###############################
    sim: DataFrame = run_kiln(kiln, ref, A1, A2, A3, A4)
    ###############################

    x: DataFrame = pd.concat(objs=[ref, sim], axis=1)
    print( x )

    rmsq: float = np.sqrt((( x['temperature'] - x['sim'] ) ** 2 ).mean())
    print( rmsq )

    if rmsq < 4.2:
        loader.plot_data( x )

    print( f"=== {count} : {A1:.2f} {A2:.2f} {A3:.2f} {A4:.2f} ===" )

    return rmsq


if __name__ == "__main__":

    parser = argparse.ArgumentParser( description="Kiln Simulator" )
    parser.add_argument( "timebase", type=str, help="The time basis. This is FROMTIME_TOTIME. Look in the data directory!" )
    parser.add_argument("--start", type=str, help="Start time for the data selection (default: start time from timebase)")
    parser.add_argument("--end", type=str, help="End time for the data selection (default: end time from timebase)")
    args = parser.parse_args()

    reference:DataFrame = loader.load_data( args.timebase )
    if args.start and args.end:
        reference = reference.loc[args.start, args.end]

    #a1 = -1.674
    a1 = 0
    #a2 = 1.61
    a2 = 1.597
    #a3 = 8.992e-12
    a3 = 0
    #a4 = 0.0000003
    a4 = 3.188e-7
    #a4 = 0

    bounds = [ (None,None), (0,2), (0,0.001), (0,0.000001) ]
    guess = [ a1, a2, a3, a4 ]

    # run_and_concat( reference, a1, a2, a3, a4 )
    # exit()

    # res = minimize(lambda x: run_and_concat(
    #               reference,
    #               x[0], x[1], x[2], x[3]),
    #               guess,
    #               bounds=bounds )
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping( lambda x: run_and_concat( reference, x[0], x[1], x[2], x[3] ),
                                guess,
                                minimizer_kwargs=minimizer_kwargs,
                                niter=100 )
    # -1.2336569678200335 1.5968261689238443 8.993482334815809e-12 3.2888993312988196e-07
    # -1.6737601492964262 1.5969736282670184 2.1522776814502484e-06 3.15861392097465e-07
    # BFGS finally: [-1.23365697e+00  1.59688647e+00  8.99213712e-12  3.18842235e-07]
    #                4.12095795950739

    print( res )
    print( dir( res ) )
    print( res.x )
    print( res.fun )
