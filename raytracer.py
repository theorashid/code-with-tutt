# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:54:51 2017

@author: tar15

Optical ray tracer using object oriented programming
Investigate performances of simple lenses
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

class Ray:
    '''
    raytracer
    Create an optical ray
    
    kwargs:
        p -- ray position, default zero vector
        k -- ray direction, default zero vector
    
    fns:
        p() -- calls the position of the ray
        k() -- calls the direction of the ray
        k_norm() -- calls the normalised direction of the ray
        append(p_in, k_in) -- add new point to list of coordinates
        vertices() -- return array of the ray path coordinates
    '''
    def __init__(self, p = [0, 0, 0], k = [0, 0, 0]): # p = position, k = direction
        '''
        Initialises the ray using an initial position and direction

        kwargs:
            p -- intial position, list of 3 elements
            k -- direction vector, list of 3 elements
    
        Exception raised if list does not equal 3
        '''
        
        if len(p) != 3:
            raise Exception("Incorrect size")
            print len(p)
        if len(k) != 3:
            raise Exception("Incorrect size")
            print len(k)
            
        p = np.array(p)
        k = np.array(k)
        
        self.__p = [p]
        self.__k = k
        
    def __repr__(self):
        return "%s(p=%s, k=%s)" % ("Ray", self.__p, self.__k)

    def __str__(self):
        return "(position vector, p: %s, direction vector, k: %s)" % (self.__p, self.__k)
        
    def p(self):
        '''
        Calls most recent position vector of ray
        '''
        return self.__p[-1]
              
    def k(self):
        '''
        Calls most recent direction vector of ray
        '''
        return self.__k
        
    def k_norm(self):
        '''
        Calls normalised most recent direction vector of ray
        '''
        return self.__k/np.linalg.norm(self.__k)
        
    def append(self, p_in, k_in): # input position, k_in, output position, k_out
        '''
        Appends new position array to list of previous position coordinates
        Does not alter direction vector, k

        kwargs:
            p -- new position vector, list of 3 elements
            k -- new direction vector, list of 3 elements
        '''
        self.__p.append(p_in) # list of arrays
        self.__k = k_in # array
        
    def vertices(self):
        '''
        Returns array of arrays (able to plot) of all previous position vectors
        '''
        return np.array(self.__p) # array of arrays
        
def snell_law(k_norm, n_norm, n_1, n_2):
    '''
    Returns normalised direction vector following refraction of ray upon intercept with a surface
    If total internal reflection occurs, None is returned

    kwargs:
        k_norm -- normalised direction vector of ray before refraction, array of 3 elements
        n_norm -- normalised normal vector to the surface, array of 3 elements
        n_1 -- refractive index of medium before refraction, float-type
        n_2 -- refractive index of medium after refraction, float-type
    '''
    cos_theta_1 = abs(np.dot(k_norm, n_norm))
    sin_theta_1 = np.sqrt(1 - cos_theta_1**2)
    if sin_theta_1 > n_2/n_1:
        return None
    sin_theta_2 = (n_1/n_2)*sin_theta_1
    cos_theta_2 = np.sqrt(1 - sin_theta_2**2)
    k_new = (n_1/n_2)*k_norm + ((n_1/n_2)*cos_theta_1 - cos_theta_2)*n_norm
    return k_new/np.linalg.norm(k_new)
        
class OpticalElement:
    '''
    Created so all derived classes have a propagate_ray method
    
    fns:
        propagate_ray() -- for all derived classes
    ''' 
    def propagate_ray(self, ray):
        '''
        propagate a ray through the optical element

        kwargs:
            ray -- ray type input from class Ray
        '''
        raise NotImplementedError()

class SphericalRefraction(OpticalElement):
    '''
    raytracer
    Creates a spherical surface optical element 

    kwargs:
        z_0 -- point of spherical surface on z-axis, float-type
        n_1 -- refractive index of initial medium, float-type
        n_2 -- refractive index of medium after refraction, float-type
        curv -- curvature of the lens, with a negative value denoting projection of curve in positive z, float-type
        a_r -- aperture radius, float-type
        
    fns:
        intercept() -- returns intercept of ray and sphere
        propagate_ray() -- propagate a ray through the sphere
    '''
    
    def __init__(self, z_0, curv, n_1, n_2, a_r):
        '''
        Initialises the spherical surface

        kwargs:
            z_0 -- point of spherical surface on z-axis, float-type
            n_1 -- refractive index of initial medium, float-type
            n_2 -- refractive index of medium after refraction, float-type
            curv -- curvature of the lens, with a negative value denoting projection of curve in positive z, float-type
            a_r -- aperture radius, float-type
        '''
        
        self.__z_0 = float(z_0)        # intercept of surface with z-axis
        self.__curv = float(curv)      # curvature
        self.__n_1 = float(n_1)        # refractive indices
        self.__n_2 = float(n_2)
        self.__a_r = float(a_r)        # aperture radius
        
    def intercept(self, ray):
        '''
        Returns the intercept vector of the ray with the spherical surface
        If there is no intercept, None is returned

        kwargs:
            ray -- ray type input from class Ray
        '''
        r_c = abs(1/self.__curv) # radius of curvature
        
        if self.__curv > 0:
            z = [0, 0, self.__z_0 + r_c] # centre of curvature
        elif self.__curv < 0:
            z = [0, 0, self.__z_0 - r_c]
        else: # self.__curv == 0:
            z = [0, 0, self.__z_0]
            
        z = np.array(z)
        r = ray.p() - z # displacement vector from ray start to centre of circle
        
        if self.__curv == 0:
            l = np.dot(r, [0, 0, 1])/np.dot(ray.k_norm(), [0, 0, 1]) # normal to plane = [0, 0, 1]
            intercept = ray.p() + l*ray.k_norm()
            
            if intercept[0]*intercept[0] + intercept[1]*intercept[1] > self.__a_r**2:
                return None
                
            else:
                return intercept
                
        else:
            l_1 = -np.dot(r, ray.k_norm()) + np.sqrt((np.dot(r, ray.k_norm()))**2 - (abs(np.dot(r, r)) - r_c**2))
            l_2 = -np.dot(r, ray.k_norm()) - np.sqrt((np.dot(r, ray.k_norm()))**2 - (abs(np.dot(r, r)) - r_c**2))
        
            l = [l_1, l_2]

            if self.__curv > 0:
                intercept = ray.p() + min(l)*ray.k_norm()
                if intercept[0]*intercept[0] + intercept[1]*intercept[1] > self.__a_r**2:
                    return None
                
                else:  # includes == case, where intersection can occur
                    return intercept
            elif self.__curv < 0:
                intercept = ray.p() + max(l)*ray.k_norm()
                if intercept[0]*intercept[0] + intercept[1]*intercept[1] > self.__a_r**2:
                    return None
                else:  
                    return intercept
                 
    def propagate_ray(self, ray):
        '''
        propagate a ray through the optical element
        Use of snells_law function to find direction vector after refraction
        Returns ray, with position at intercept and direction vector after refraction

        If there is no intercept, None is returned
        If total internal reflection occurs, None is returned

        kwargs:
            ray -- ray type input from class Ray
        '''
        intercept = self.intercept(ray)
        if intercept is None:
            return "No intercept"
        else:
            z = self.__z_0 + 1/self.__curv # z coordinate of centre of sphere
            normal = np.array([intercept[0], intercept[1], intercept[2] - z])
            n_norm = normal/np.linalg.norm(normal)
            k_r_norm = snell_law(ray.k_norm(), n_norm, self.__n_1, self.__n_2)
            if k_r_norm is None:
                return "Total internal reflection"
            else:
                ray.append(intercept, k_r_norm)
                return ray
      
class PlanoConvexLens(OpticalElement):
    '''
    raytracer
    Creates a planoconvex lens optical element 

    kwargs:
        z_0 -- point of spherical surface on z-axis, float-type
        n_1 -- refractive index of initial medium, float-type
        n_2 -- refractive index of medium after refraction, float-type
        sepn -- separation between z_0 and the plane surface, float-type
        curv -- curvature of the lens, with a negative value denoting projection of curve in positive z, float-type
        
    fns:
        propagate_ray() -- propagate a ray through the lens
    '''
    
    def __init__(self, z_0, n_1, n_2, sepn, curv):
        '''
        Initialises the spherical surface

        kwargs:
            z_0 -- point of spherical surface on z-axis, float-type
            n_1 -- refractive index of initial medium, float-type
            n_2 -- refractive index of medium after refraction, float-type
            sepn -- separation between z_0 and the plane surface, float-type
            curv -- curvature of the lens, with a negative value denoting projection of curve in positive z, float-type
        
        If there is no separation, None is returned
        '''
        self.__z_0 = float(z_0)
        self.__n_1 = float(n_1)        
        self.__n_2 = float(n_2)
        self.__sepn = float(abs(sepn))
        self.__curv = float(curv)
        
        if self.__curv == 0:
            return None
        
        else:
            r_c = abs(1/self.__curv)
            self.__a_r = np.sqrt(2*r_c*self.__sepn - self.__sepn**2) # aperture radius
            
            if self.__curv > 0:
                self.__spherical = SphericalRefraction(self.__z_0, n_1, n_2, self.__a_r, self.__curv)                                        
                self.__plane = SphericalRefraction(self.__z_0 + self.__sepn, n_2, n_1, self.__a_r, 0.0)
            else:
                self.__plane = SphericalRefraction(self.__z_0, n_1, n_2, self.__a_r, 0.0)
                self.__spherical = SphericalRefraction(self.__z_0 + self.__sepn, n_2, n_1, self.__a_r, self.__curv)                                        

    def __str__(self):
        if self.__curv > 0:
            return "Spherical surface at = %s, plane at z = %s" %(self.__z_0, self.__z_0 + self.__sepn)
        if self.__curv < 0:
            return "Plane at z = %s, spherical surface at = %s" %(self.__z_0, self.__z_0 + self.__sepn)
        
    def propagate_ray(self, ray):
        '''
        propagate a ray through the optical element
        Use of SphericalRefraction class to find direction vector after refraction
        Returns ray, with position at intercept and direction vector after refraction
        
        If the curvature is 0, as before, None is returned
        
        If there is no intercept, None is returned
        If total internal reflection occurs, None is returned

        kwargs:
            ray -- ray type input from class Ray
        '''                        
        if self.__curv == 0:
            return None
        elif self.__curv > 0:
            SphericalRefraction.propagate_ray(self.__spherical, ray)
            SphericalRefraction.propagate_ray(self.__plane, ray)
        else: # self.__curv < 0
            SphericalRefraction.propagate_ray(self.__plane, ray)
            SphericalRefraction.propagate_ray(self.__spherical, ray)

class OutputPlane(OpticalElement):
    '''
    raytracer
    Creates an output plane at a desired z-coordinate

    kwargs:
        z -- z-coordinate at which plane is created, float-type
    
    fns:
        intercept() -- returns intercept vector of ray with output plane
        propagate_ray() -- propagates ray through output plane
    '''
    
    def __init__(self, z):
        
        """
        Initialises the output plane at a chosen z-coordinate
        
        kwargs:
            z -- z-coordinate of the output plane
        """ 
        
        self.__z = float(z)
        self.__norm_n = np.array([0., 0., 1.])
        
    def __str__(self):
        
        return "Plane at z = %s, normal = %s" %(self.__z, self.__norm_n)
        
    def intercept(self, ray):
        '''
        Returns the intercept vector of the ray with the plane
        If there is no intercept, None is returned
    
        If there is no intercept (ray travels parallel to plane), None is returned

        kwargs:
            ray -- ray type input from class Ray
        '''
        r = np.array([0., 0., self.__z]) - ray.p()
        if np.dot(ray.k_norm(), self.__norm_n) != 0:
            l = np.dot(r, self.__norm_n)/np.dot(ray.k_norm(), self.__norm_n)
            intercept = ray.p() + l*ray.k_norm()
            return intercept
        else:
            return None
    
    def propagate_ray(self, ray):
        '''
        propagate a ray through the plane
        Returns ray, with position at intercept and direction vector
        
        kwargs:
            ray -- ray type input from class Ray
        '''
        ray.append(self.intercept(ray), ray.k())
        
def rtpairs(R, N): # R for radii, r. N for angles, theta
    '''
    Generates a sequence of r,theta coordinate pairs
        
    kwargs:
        R -- list of radii, list
        N -- list of number of angles for each radius, list
    '''
    for i in N:
        r = R[N.index(i)]
        theta = np.linspace(0, 2*np.pi, i, endpoint = False) # endpoint as values at 0 and 2pi are equal
        for x in theta:
            yield (r, x)
    
def rtuniform(n, R, m):
    '''
    Generates a sequence of r, theta pairs that are uniformly distributed along a disc
    
    kwargs:
        n -- number of rings, int-type
        R -- maximum radius of the disc, float-type
        m -- scaling factor of the rings, int-type
    '''
    radii = list(np.linspace(0, R, n+1))
    npoints = list(np.linspace(0, m*n, n+1))
    npoints = [1 if x == 0 else x for x in npoints]
    return rtpairs(radii, npoints)
    
class BundleRays:
    '''
    raytracer
    Create a bundle of rays
    
    kwargs:
        n -- number of concentric rings, int-type
        R -- radius of the outermost ring, float-type
        m -- scaling factor of the rings, int-type
        k -- communal direction vector of the rays, list
    
    fns:
        propagate_ray(optical_elements) -- propagates rays through a list of optical elements
        plot_x_z() -- Plots track of the ray along x and z axes
        plot_x_y() -- Plots the cross section in x-y plane of the ray bundle at the output plane
        
    '''
    def __init__(self, n, R, m, k = [0, 0, 1]):
        '''
        Initialises the rays using basic parameters and direction vector

        kwargs:
            n -- number of concentric rings, float-type
            R -- radius of the outermost ring, float-type
            m -- scaling factor of the rings, float-type
            k -- communal direction vector of the rays, list of 3 elements, default [0, 0, 1]
    
        Exception raised if list does not equal 3
        '''
        if len(k) != 3:
            raise Exception("Incorrect size")
            print len(k)
        
        self.__ray_list = []
        self.__n = n
        self.__R = R
        self.__m = m
        for (r, t) in rtuniform(n, R, m):
            self.__ray_list.append(Ray([r*np.cos(t), r*np.sin(t), 0], k))
            
    def __str__(self):
        return "Bundle of rays, radius = %s, m = %s and direction vector = %s" %(self.__R, self.__m, self.__k)
    
    def propagate_rays(self, optical_elements = []):
        """
        Propagates bundle of rays through several optical elements
        
        kwargs:
           optical_elements -- empty list to be filled with optical elements 
        """
        for optical_element in optical_elements:
            for i in self.__ray_list:
                optical_element.propagate_ray(i)
                
    def plot_ray_x_z(self):
        """
        Plots track of the ray along x and z axes
        """ 
        for ray in self.__ray_list:
            plt.plot([z[2] for z in ray.vertices()], [x[0] for x in ray.vertices()], 'r-')
        plt.xlabel('z/mm')
        plt.ylabel('x/mm')
        plt.title("Track of rays in x-z plane")
        plt.xlim(0, 500)   
        plt.ylim(-10.0, 10.0)
        plt.show()
        
    def plot_ray_x_y(self, optical_element):
        """
        Plots the cross section in x-y plane of the ray bundle at the focus plane
        Focus plane must be penultimate optical element
        
        kwargs:
            optical_element -- position of optical element in sequence, int-type
        """ 
        for ray in self.__ray_list:
            plt.plot(ray.vertices()[optical_element,0], ray.vertices()[optical_element,1], 'ro')
        plt.xlabel('x/mm')
        plt.ylabel('y/mm')
        if optical_element == 0:
            plt.title("z = 0")
        if optical_element == -2: # usually focus plane, before output plane
            plt.title("z = focus")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def root_mean_square(self):
        '''
        for focus, last optical element should be focus output_plane
        '''
        radius_x_square = []
        radius_y_square = []
        radius_square = []
        for ray in self.__ray_list:
            radius_x_square.append((ray.vertices()[-1])[0]**2)
            radius_y_square.append((ray.vertices()[-1])[1]**2)
        for i in range(len(radius_x_square)):
            square = radius_x_square[i] + radius_y_square[i]
            radius_square.append(square)
        rms = np.sqrt(sum(radius_square)/len(radius_x_square))
        return rms

def paraxial_focus(optical_elements = []):
    '''
    returns z coordinate of focus
    
    kwargs:
           optical_elements -- empty list to be filled with optical elements
    '''
    test_ray = Ray(p = [0.01, 0., 0.], k = [0, 0, 1])
    for optical_element in optical_elements:
        optical_element.propagate_ray(test_ray)
    
    l = - test_ray.p()[0]/test_ray.k()[0]    
    axis_intercept = test_ray.p() + l*test_ray.k()
    return axis_intercept[2]

# Experiment set up for investigation into spherical aberration of spherical surface
sphere = SphericalRefraction(100.0, 0.02, 1.0, 1.15168, 100.0)
output_plane = OutputPlane(500)
focus = paraxial_focus([sphere])
focal_length = focus - 100.0
print focal_length
output_focus = OutputPlane(focus)

R_values = np.linspace(0.1, 10.0, 100)
bundles_of_rays = []
rms_radii = []

for i in R_values:
    bundles_of_rays.append(BundleRays(10, i, 5))
    
for bundle in bundles_of_rays:
    bundle.propagate_rays([sphere, output_focus])
    rms_radii.append(bundle.root_mean_square())
    
geometric_spot_size = [np.pi*(i**2) for i in rms_radii]
# rms_radii_root = [np.sqrt(i) for i in rms_radii]

plt.plot(R_values, rms_radii, 'b-')
plt.xlabel('R, radius of bundle/mm')
plt.ylabel('rms radius at focus/mm')
plt.xlim(0, 11)   
plt.ylim(0, 0.10)
plt.title("Graph of root mean square radius at focus against a_r")
plt.grid(1)
plt.show()

def fit(x, a, b, c): # quadratic fitting function
    return a*x**2 + b*x + c
    
initial_guess = [0.1, 0.0, 0.0] # initial guess based on graph observation
po, po_cov = spo.curve_fit(fit, R_values, rms_radii, initial_guess)
print po[0], po[1], po[2]

plt.plot(R_values, geometric_spot_size, 'b-')
plt.xlabel('R, radius of bundle/mm')
plt.ylabel('geometric spot size at focus/mm^2')
plt.xlim(0, 11)   
plt.ylim(0, 0.025)
plt.title("Graph of geometric spot size at focus against a_r")
plt.grid(1)
plt.show()
'''
bundle_rays_min_max = [BundleRays(10, 0.1, 5), BundleRays(10, 10, 5)]
for bundle in bundle_rays_min_max:
    bundle.propagate_rays([sphere, output_focus, output_plane])
    bundle.plot_ray_x_z()
    bundle.plot_ray_x_y(-2)
'''
bundle_rays_max = BundleRays(10, 10, 5)
bundle_rays_max.propagate_rays([sphere, output_focus, output_plane])
bundle_rays_max.plot_ray_x_z()
bundle_rays_max.plot_ray_x_y(0)
bundle_rays_max.plot_ray_x_y(-2)

wavelength_max = []
for i in range(len(R_values)):
    wavelength = R_values[i]*rms_radii[i]*1e6/focal_length
    wavelength_max.append(wavelength)
    
plt.plot(R_values, wavelength_max, 'b-')
plt.xlabel('R, radius of bundle/mm')
plt.ylabel('max wavelength of no visible spherical aberration/nm')
plt.xlim(0, 11)   
plt.ylim(0, 2500)
plt.title("Graph of wavelength against a_r")
plt.grid(1)
plt.show()