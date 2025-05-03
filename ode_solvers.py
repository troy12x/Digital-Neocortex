"""
Specialized ODE solvers for the Digital Neocortex architecture.
These solvers are optimized for the continuous-time neural dynamics used in the model.
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint


class EulerSolver:
    """
    Simple Euler method for solving ODEs.
    Fast but less accurate than higher-order methods.
    """
    def __init__(self, step_size=0.1):
        self.step_size = step_size
    
    def solve(self, func, y0, t, args=None):
        """
        Solve the ODE system defined by func.
        
        Args:
            func: Function defining the ODE system (dy/dt = func(t, y))
            y0: Initial state
            t: Time points at which to return the solution
            args: Additional arguments to pass to func
            
        Returns:
            Solution at the requested time points
        """
        device = y0.device
        dtype = y0.dtype
        
        # Initialize solution array
        solution = torch.zeros((len(t),) + y0.shape, dtype=dtype, device=device)
        solution[0] = y0
        
        # Solve using Euler method
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            steps = max(1, int(dt / self.step_size))
            step_size = dt / steps
            
            y = solution[i-1]
            
            # Take multiple small steps to reach the next time point
            for _ in range(steps):
                if args is not None:
                    dy = func(t[i-1], y, *args)
                else:
                    dy = func(t[i-1], y)
                y = y + step_size * dy
            
            solution[i] = y
        
        return solution


class RungeKutta4Solver:
    """
    4th-order Runge-Kutta method for solving ODEs.
    Good balance between accuracy and computational cost.
    """
    def __init__(self, step_size=0.1):
        self.step_size = step_size
    
    def solve(self, func, y0, t, args=None):
        """
        Solve the ODE system defined by func.
        
        Args:
            func: Function defining the ODE system (dy/dt = func(t, y))
            y0: Initial state
            t: Time points at which to return the solution
            args: Additional arguments to pass to func
            
        Returns:
            Solution at the requested time points
        """
        device = y0.device
        dtype = y0.dtype
        
        # Initialize solution array
        solution = torch.zeros((len(t),) + y0.shape, dtype=dtype, device=device)
        solution[0] = y0
        
        # Solve using RK4 method
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            steps = max(1, int(dt / self.step_size))
            step_size = dt / steps
            
            y = solution[i-1]
            
            # Take multiple small steps to reach the next time point
            for _ in range(steps):
                if args is not None:
                    k1 = func(t[i-1], y, *args)
                    k2 = func(t[i-1] + step_size/2, y + step_size/2 * k1, *args)
                    k3 = func(t[i-1] + step_size/2, y + step_size/2 * k2, *args)
                    k4 = func(t[i-1] + step_size, y + step_size * k3, *args)
                else:
                    k1 = func(t[i-1], y)
                    k2 = func(t[i-1] + step_size/2, y + step_size/2 * k1)
                    k3 = func(t[i-1] + step_size/2, y + step_size/2 * k2)
                    k4 = func(t[i-1] + step_size, y + step_size * k3)
                
                y = y + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            solution[i] = y
        
        return solution


class AdaptiveStepSolver:
    """
    Adaptive step size solver for ODEs.
    Adjusts step size based on error estimates for better accuracy and efficiency.
    """
    def __init__(self, rtol=1e-3, atol=1e-6, min_step=1e-6, max_step=1e-1):
        self.rtol = rtol
        self.atol = atol
        self.min_step = min_step
        self.max_step = max_step
    
    def solve(self, func, y0, t, args=None):
        """
        Solve the ODE system using adaptive step size control.
        
        Args:
            func: Function defining the ODE system (dy/dt = func(t, y))
            y0: Initial state
            t: Time points at which to return the solution
            args: Additional arguments to pass to func
            
        Returns:
            Solution at the requested time points
        """
        # For complex systems, we use torchdiffeq's odeint which has built-in
        # adaptive step size control
        return odeint(func, y0, t, rtol=self.rtol, atol=self.atol, 
                      method='dopri5', args=args)


class ParallelODESolver:
    """
    Solver that processes multiple independent ODE systems in parallel.
    Useful for spatial units in the Space-Time Layer.
    """
    def __init__(self, solver='rk4', step_size=0.1, rtol=1e-3, atol=1e-6):
        self.solver_type = solver
        self.step_size = step_size
        self.rtol = rtol
        self.atol = atol
        
        if solver == 'euler':
            self.solver = EulerSolver(step_size)
        elif solver == 'rk4':
            self.solver = RungeKutta4Solver(step_size)
        elif solver == 'adaptive':
            self.solver = AdaptiveStepSolver(rtol, atol)
        else:
            raise ValueError(f"Unknown solver type: {solver}")
    
    def solve_batch(self, funcs, y0_batch, t, args_batch=None):
        """
        Solve multiple ODE systems in parallel.
        
        Args:
            funcs: List of functions defining the ODE systems
            y0_batch: Batch of initial states [batch_size, ...]
            t: Time points at which to return the solution
            args_batch: Batch of additional arguments to pass to funcs
            
        Returns:
            Batch of solutions at the requested time points
        """
        batch_size = y0_batch.shape[0]
        device = y0_batch.device
        dtype = y0_batch.dtype
        
        # Initialize solution array
        solution_shape = (len(t), batch_size) + y0_batch.shape[1:]
        solutions = torch.zeros(solution_shape, dtype=dtype, device=device)
        
        # Solve each system independently
        for b in range(batch_size):
            y0 = y0_batch[b]
            func = funcs[b] if isinstance(funcs, list) else funcs
            
            if args_batch is not None:
                args = args_batch[b] if isinstance(args_batch, list) else args_batch
                solution = self.solver.solve(func, y0, t, args)
            else:
                solution = self.solver.solve(func, y0, t)
            
            solutions[:, b] = solution
        
        return solutions


def create_solver(solver_type='dopri5', step_size=0.1, rtol=1e-3, atol=1e-6):
    """
    Factory function to create an appropriate ODE solver.
    
    Args:
        solver_type: Type of solver ('euler', 'rk4', 'dopri5', etc.)
        step_size: Step size for fixed-step methods
        rtol: Relative tolerance for adaptive methods
        atol: Absolute tolerance for adaptive methods
        
    Returns:
        ODE solver instance
    """
    if solver_type == 'euler':
        return EulerSolver(step_size)
    elif solver_type == 'rk4':
        return RungeKutta4Solver(step_size)
    elif solver_type == 'adaptive':
        return AdaptiveStepSolver(rtol, atol)
    elif solver_type == 'parallel':
        return ParallelODESolver('rk4', step_size, rtol, atol)
    else:
        # Default to torchdiffeq's odeint with the specified method
        def solver(func, y0, t, args=None):
            return odeint(func, y0, t, method=solver_type, 
                          rtol=rtol, atol=atol, args=args)
        return solver
