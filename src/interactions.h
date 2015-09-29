#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 calculateRandomDirectionSpecular(glm::vec3 normal, glm::vec3 incident, thrust::default_random_engine &rng, float n) {
	thrust::uniform_real_distribution<float> u(0, 1);
	float r1 = u(rng);
	float r2 = u(rng);
	float theta = acos(pow(r1, (1.0f / (n + 1.0f))));
	float phi = 2.0f * PI * r2;
	//^^^ a vector of jitter
	//the perfect specular direction:
	glm::vec3 dspec = -(2.0f * glm::dot(normal, incident) * normal - incident);

	glm::vec3 directionNotNormal;
	if (abs(dspec.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(dspec.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(dspec, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(dspec, perpendicularDirection1));

	return dspec
		+ cos(phi)*sin(theta) * perpendicularDirection1
		+ sin(phi)*sin(theta) * perpendicularDirection2;
}

__host__ __device__
glm::vec3 calculateRefraction(glm::vec3 normal, glm::vec3 incident, bool outside, float ior1, float ior2) {
	float sint1 = incident.x*incident.x + incident.y*incident.y;
	float ior = ior2 / ior1;
	float sint2 = ior*ior*sint1;
	if (sint2 >= 1.0f && !outside) { return glm::vec3(0); }
	float cost = sqrtf(max(0.f, 1.f - sint2));
	if (outside) cost = -cost;

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return cost * normal
		+ ior * -incident.x * perpendicularDirection1
		+ ior * -incident.y * perpendicularDirection2;
}
/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * - (NOT RECOMMENDED - converges slowly or badly especially for pure-diffuse
 *   or pure-specular. In principle this correct, though.)
 *   Always take a 50/50 split between a diffuse bounce and a specular bounce,
 *   but multiply the result of either one by 1/0.5 to cancel the 0.5 chance
 *   of it happening.
 * - Pick the split based on the intensity of each color, and multiply each
 *   branch result by the inverse of that branch's probability (same as above).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
        Ray &ray,
		bool outside,
        glm::vec3 &color,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1); 
	float randoCalrisian = u01(rng); //here's our random number. based on it, choose what to do!
	float intensityDiff = m.color.x + m.color.y + m.color.z;
	float intensitySpec = m.specular.color.x + m.specular.color.y + m.specular.color.z;
	float schlick = 1.0f;
	bool tir = false;
	if (m.hasRefractive) {
		float r0 = pow((1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction), 2);
		schlick = r0 + (1 - r0)*pow(1.0f - glm::dot(normal, -ray.direction), 5);
		//printf("%f \n", schlick);
	}
	if (!outside) {
		float critangle = asin(1.0f / m.indexOfRefraction);
		tir = glm::dot(ray.direction,normal) < critangle; 
	}
	float Diff = intensityDiff / (intensityDiff + intensitySpec);
	float Spec = intensitySpec / (intensityDiff + intensitySpec);
	if (randoCalrisian < Diff && !m.hasRefractive) {//diffuse
		ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		ray.origin = intersect;
		color *= m.color/Diff;
	}
	else if (randoCalrisian < Diff + Spec){//specular - transmit or reflect!
		if (!m.hasRefractive || randoCalrisian < schlick) {//reflect
			ray.direction = calculateRandomDirectionSpecular(normal, ray.direction, rng, m.specular.exponent);
			ray.origin = intersect;
			color *= m.specular.color / Spec;
		}
		else {//transmit! transmission is 1 - reflectivity according to fresnel
			float ior = m.indexOfRefraction;
			if (outside) {
				ior = 1.f / ior;
			}
			ray.direction = glm::refract(ray.direction, normal, ior);//calculateRefraction(normal,ray.direction,outside,ior1,ior2);
			ray.origin = intersect + ray.direction*0.001f; //this should push origins across refractive interface?
			color *= m.color;
		}
	}
}
