/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::sin;
using std::cos;

// Random number generator: this is global.  
std::default_random_engine g_gen;

// helper for the error calculation
double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {


  if (!is_initialized)
  {
    num_particles = 250;  // Number of particles
    normal_distribution<double> dist_x(x, std[0]);  
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    particles.resize(num_particles);
    weights.resize(num_particles);
    for (int i=0;i<num_particles;i++)
    {
      Particle &p = particles[i];
      p.id = i;
      p.x = dist_x(g_gen);  
      p.y = dist_y(g_gen);  
      p.theta = dist_theta(g_gen);  
      p.weight = 1.;          
      weights[i] = p.weight;     
    }
    is_initialized = true;
  } else {
    std::cerr << "Reinit ? This method should be called only once." << std::endl;
    exit(-1);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
 
    for (int i=0;i<num_particles;i++)
    {
      Particle &p = particles[i];

      if (yaw_rate<0.0001)
      {
        // avoid division by zero:
        // if yaw_rate is almost zero, the car is going straight ahead 
        p.x = p.x + velocity*delta_t*cos(p.theta);
        p.y = p.y + velocity*delta_t*sin(p.theta);
        p.theta+=yaw_rate*delta_t;
      }
      else
      {
        p.x = p.x + (velocity / yaw_rate) * (sin( p.theta + yaw_rate*delta_t ) - sin (p.theta));
        p.y = p.y + (velocity / yaw_rate) * (cos (p.theta) - cos( p.theta + yaw_rate*delta_t ) );
        p.theta+=yaw_rate*delta_t;
      }
      
      normal_distribution<double> dist_x(0, std_pos[0]);  
      normal_distribution<double> dist_y(0, std_pos[1]);
      normal_distribution<double> dist_theta(0, std_pos[2]);
      p.x += dist_x(g_gen);
      p.y += dist_y(g_gen);
      p.theta += dist_theta(g_gen);
    }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  for (size_t i=0;i<observations.size();i++)
  {
    LandmarkObs &obs = observations[i];
    
    double mindist = std::numeric_limits<const double>::infinity();
    int closest_lm_id = -1;

    for (size_t j=0;j<predicted.size();j++)
    {
      LandmarkObs &real_lm = predicted[j];
      double dx = real_lm.x - obs.x;
      double dy = real_lm.y - obs.y;
      double dist = sqrt( dx * dx + dy * dy );
      if (mindist > dist) {
        mindist = dist; 
        closest_lm_id = real_lm.id;
      }
    }

    if (closest_lm_id != -1)
    {
      obs.id = closest_lm_id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
    // Loop, for all particles
    for (int i=0;i<num_particles;i++)
    {
      // Step 1: For the given particle: Collect predicted observation array from the Map, using the given particle location. ("What we should have seen ?")
      Particle &p = particles[i];
      vector<LandmarkObs> predicted;
      for (size_t lmidx = 0;lmidx<map_landmarks.landmark_list.size();lmidx++)
      {
        const Map::single_landmark_s &lm = map_landmarks.landmark_list[lmidx];
        double dx = p.x - lm.x_f;
        double dy = p.y - lm.y_f;
        double dist = sqrt( dx * dx + dy * dy );
        if (dist < sensor_range )
        {
          predicted.push_back( { lm.id_i, lm.x_f, lm.y_f } );  
        }
      }

      // Step 2: For the given particle: Convert the sense values from the vehicle coordinate system to the map's coord. system ("Where did we sense landmarks on the map ?")
      vector<LandmarkObs> observed;

      double costheta = cos( p.theta );
      double sintheta = sin( p.theta );
      size_t num_obs = observations.size();
      p.sense_x.resize(num_obs);
      p.sense_y.resize(num_obs);
      for (size_t j=0;j<num_obs;j++)
      {
        const LandmarkObs &obs = observations[j];
        p.sense_x[j] = costheta * obs.x - sintheta * obs.y + p.x;
        p.sense_y[j] = sintheta * obs.x + costheta * obs.y + p.y;
        observed.push_back( { -1, p.sense_x[j], p.sense_y[j] } );
      }

      // Step 3: For the given particle: Match the observations with the actual landmarks ("Which landmarks did we see ?")
      dataAssociation( predicted, observed );
      vector<int> assoc;
      assoc.resize(num_obs);
      for (size_t j=0;j<num_obs;j++)
      {
        assoc[j] = observed[j].id;
      }
      SetAssociations(p, assoc, p.sense_x, p.sense_y);

      // Step 4: Calculate weights ("How close is this particle to the car's real location/heading ?")
      double final_weight = 1;
      for (size_t j=0;j<num_obs;j++)
      {
        const LandmarkObs &obs = observed[j];
        if (obs.id!=-1)
        {
          final_weight = final_weight * multiv_prob( std_landmark[0], std_landmark[1], obs.x, obs.y, map_landmarks.landmark_list[obs.id-1].x_f, map_landmarks.landmark_list[obs.id-1].y_f);
        } else {
          // This can only happen if there are no landmarks on the map
          std::cerr << "Fatal error: The map is empty !" << std::endl;
          exit(-1);
        }
      }
      p.weight = final_weight;
       
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Step 1: Calculate the normalized weights and put it in the "weights" vector
  double sum = 0;
  size_t num_particles = particles.size();
  for (size_t i = 0; i < num_particles; i++) sum += particles[i].weight;
  double maxw = 0;
  for (size_t i = 0; i < num_particles; i++) 
  {
    particles[i].weight/=sum;
    weights[i] = particles[i].weight;
    maxw = std::max(maxw, weights[i]);
  }

  if (maxw == 0)
  {
    std::cerr << "Fatal error, something is wrong here." << std::endl;
    exit(-1); 
  }

  // Step 2: Implement the resampling wheel algorithm and create the new particle generation
  vector<Particle> pold = particles;
  int index = g_gen() % num_particles;
  double beta = 0.0;
  std::uniform_real_distribution<double> uni_dist(0.0,2.0*maxw);
  for (size_t i = 0; i < num_particles; i++)
  {
    beta += uni_dist(g_gen);      
    while (beta > weights[index]) 
    { 
      beta -= weights[index];
      index = (index + 1) % num_particles;
    } 

    particles[i] = pold[index];
    particles[i].id = i;
  }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}