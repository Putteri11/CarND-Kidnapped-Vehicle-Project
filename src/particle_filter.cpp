/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
  double std_x, std_y, std_theta;

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  int n = 200;
  num_particles = n;

  weights.clear();
  weights.assign(n, 1.0);
  particles.clear();

  for (int i = 0; i < n; i++) {
    double sample_x, sample_y, sample_theta;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    Particle particle;
    
    particle.x = sample_x;
    particle.y = sample_y;
    particle.theta = sample_theta;
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
 
  double std_x, std_y, std_theta;

  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];

  default_random_engine gen;
  double th = 0.01;
  
  normal_distribution<double> dist_x(0.0, std_x);
  normal_distribution<double> dist_y(0.0, std_y);
  normal_distribution<double> dist_theta(0.0, std_theta);

  for (int i = 0; i < num_particles; i++) {
    Particle &p = particles[i];

    if (yaw_rate < th && yaw_rate > -th) {
      p.x += cos(p.theta) * velocity * delta_t;
      p.y += sin(p.theta) * velocity * delta_t;
    } else {
      p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += (velocity / yaw_rate) * (-cos(p.theta + yaw_rate * delta_t) + cos(p.theta)); 
    }

    p.theta += yaw_rate * delta_t;

    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);

  }

}

void ParticleFilter::dataAssociation(double sensor_range, const Map &map_landmarks, const std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;

  for (int i = 0; i < num_particles; i++) {
    associations.clear();
    sense_x.clear();
    sense_y.clear();

    Particle &p = particles[i];

    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs obs = observations[j];
      double t = p.theta;
      double x_map = p.x + cos(t) * obs.x - sin(t) * obs.y; 
      double y_map = p.y + sin(t) * obs.x + cos(t) * obs.y;

      int minID = -1;
      double minDist = INT_MAX;

      for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
	double d_particle = dist(p.x, p.y, double(map_landmarks.landmark_list[k].x_f), double(map_landmarks.landmark_list[k].y_f));
	double d = dist(x_map, y_map, double(map_landmarks.landmark_list[k].x_f), double(map_landmarks.landmark_list[k].y_f));
	if (d < minDist && d_particle < sensor_range) {
	  minID = map_landmarks.landmark_list[k].id_i;
	  minDist = d;
	}
      }

      if (minID != -1) { 
        sense_x.push_back(x_map);
        sense_y.push_back(y_map);
        associations.push_back(minID);
      } 
    }
    //cout << associations.size() << endl;
    p = SetAssociations(p, associations, sense_x, sense_y);
  }


}


double ParticleFilter::multivariateGaussian(double x, double y, double mx, double my, double stds[]) {
  double std_x = stds[0];
  double std_y = stds[1];

  return (1.0 / (2 * M_PI * std_x * std_y)) * exp( -0.5 * (pow((x - mx)/std_x, 2.0
) + pow((y - my)/std_y, 2.0)) );
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
 
 // weights.clear();
 // weights.assign(num_particles, 1.0);

  dataAssociation(sensor_range, map_landmarks, observations);

  for (int i = 0; i < num_particles; i++) {
    double w = 1.0;
    Particle &p = particles[i];
    //cout << p.associations.size() << endl;
    for (int j = 0; j < p.associations.size(); j++) {
      double x, y, mx, my;
      x = p.sense_x[j];
      y = p.sense_y[j];
      int index = -1;
      int ID = p.associations[j];
      int k = 0;
      bool found = false;

      while (k < map_landmarks.landmark_list.size() && !found) {
	if (map_landmarks.landmark_list[k].id_i == ID) {
	  index = k;
	  found = true;
	}
	k++;
      }

      mx = double(map_landmarks.landmark_list[index].x_f);
      my = double(map_landmarks.landmark_list[index].y_f);
      //cout << multivariateGaussian(x, y, mx, my, std_landmark) << endl;
      w *= multivariateGaussian(x, y, mx, my, std_landmark);
    }

    weights[i] = w;
    p.weight = w;
    //cout << w << endl; 
  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;

  discrete_distribution<> d(weights.begin(), weights.end());

  std::vector<Particle> new_particles;
  new_particles.clear();

  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[d(gen)]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
