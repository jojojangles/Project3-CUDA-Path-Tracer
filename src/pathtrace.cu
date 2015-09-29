#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
	cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	getchar();
    exit(EXIT_FAILURE);
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    return thrust::default_random_engine(utilhash((index + 1) * iter) ^ utilhash(depth));
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geo = NULL;
static Material *dev_mat = NULL;
static Ray *dev_rays = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	const int geolements = hst_scene->geoms.size();
	const int matlements = hst_scene->materials.size();
	cudaMalloc(&dev_geo, geolements * sizeof(Geom));
	cudaMalloc(&dev_mat, matlements * sizeof(Material));
	cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));
	cudaMemcpy(dev_geo, hst_scene->geoms.data(), geolements * sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat, hst_scene->materials.data(), matlements * sizeof(Material), cudaMemcpyHostToDevice);
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_geo);
	cudaFree(dev_mat);
	cudaFree(dev_rays);
    checkCUDAError("pathtraceFree");
}

__global__ void rayBuilder(Camera cam, Ray *rays, float tanx, float tany, glm::vec3 right, glm::vec3 perup, int pixelcount, int iter) {
	int uidx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int vidx = (blockIdx.y * blockDim.y) + threadIdx.y;
	int rayidx = vidx * cam.resolution.x + uidx;

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, rayidx, 1);
	thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
	float result = u01(rng);

	float u = (2.0f * (uidx + result) / cam.resolution.x - 1.0f);
	float v = (2.0f * (vidx + result) / cam.resolution.y - 1.0f);
	if (rayidx <  pixelcount) {
		glm::vec3 eye = cam.position;
		glm::vec3 pixel = eye + cam.view - tanx*u*right - tany*v*perup;
		rays[rayidx].origin = eye;
		rays[rayidx].direction = glm::normalize(pixel - eye);
		rays[rayidx].color = glm::vec3(1.0f);
		rays[rayidx].pixel = rayidx;
	}
}

__global__ void rayDebug(Camera cam, glm::vec3 *image, Ray *rays) {
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;
	int rayidx = v * cam.resolution.x + u;
	image[rayidx] += glm::abs(rays[rayidx].direction);
}

__global__ void slingRays(Ray *rays, Geom *geo, int geocount, Material *mats, glm::vec3 *image, int pixelcount, int depth, int iter) {
	int ridx = blockIdx.x * blockDim.x + threadIdx.x;
	if (ridx < pixelcount) {
		Ray r = rays[ridx];
		if (glm::length(r.direction) < 0.0001f) { return; }
		const float t0 = -1.0f;
		float t1 = INFINITY;
		bool hit = false;
		bool light = false;
		bool outside = true;
		glm::vec3 pt = glm::vec3(0.0f);
		glm::vec3 temp_pt = glm::vec3(0.0f);
		glm::vec3 nml = glm::vec3(0.0f);
		glm::vec3 temp_nml = glm::vec3(0.0f);
		Geom ghit;
		Geom g = geo[0];

		for (int i = 0; i < geocount; i++) {
			g = geo[i];
			if (g.type == SPHERE) {
				float temp = sphereIntersectionTest(g, r, temp_pt, temp_nml, outside);
				if (t0 < temp && temp < t1) {
					t1 = temp; pt = temp_pt; nml = temp_nml; hit = true; ghit = g;
				}
			}
			else if (g.type == CUBE) {
				float temp = boxIntersectionTest(g, r, temp_pt, temp_nml, outside);
				if (t0 < temp && temp < t1) {
					t1 = temp; pt = temp_pt; nml = temp_nml; hit = true; ghit = g;
				}
			}
		}

		if (hit) {
			float emit = mats[ghit.materialid].emittance;
			if (emit > 0.0f) {
				r.color *= mats[ghit.materialid].emittance;
				r.direction = glm::vec3(0.0f);
			}
			else {
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, ridx, depth);
				scatterRay(r, outside, r.color, pt, nml, mats[ghit.materialid], rng);
			}
		}
		else { //terminate
			r.color = glm::vec3(0.1f,0.1f,0.1f);
			r.direction = glm::vec3(0.0f);
		}
		rays[ridx] = r;
	}
}

__global__ void consumeRays(Ray *rays, glm::vec3 *image, int pixelcount) {
	int ridx = blockIdx.x * blockDim.x + threadIdx.x;
	if (ridx < pixelcount) {
		Ray r = rays[ridx];
		if (glm::length(r.direction) < 0.00001f) {
			image[r.pixel] += r.color;
			//printf("%f %f %f \n", r.color.x, r.color.y, r.color.z);
		}
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const int blockSideLength = 8;
    const dim3 blockSize(blockSideLength, blockSideLength);
    const dim3 blocksPerGrid(
            (cam.resolution.x + blockSize.x - 1) / blockSize.x,
            (cam.resolution.y + blockSize.y - 1) / blockSize.y);

    ///////////////////////////////////////////////////////////////////////////
    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray is a (ray, color) pair, where color starts as the
    //     multiplicative identity, white = (1, 1, 1).
    //   * For debugging, you can output your ray directions as colors.
    // * For each depth:
    //   * Compute one new (ray, color) pair along each path (using scatterRay).
    //     Note that many rays will terminate by hitting a light or hitting
    //     nothing at all. You'll have to decide how to represent your path rays
    //     and how you'll mark terminated rays.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //       surface.
    //     * You can debug your ray-scene intersections by displaying various
    //       values as colors, e.g., the first surface normal, the first bounced
    //       ray direction,  the first unlit material color, etc.
    //   * Add all of the terminated rays' results into the appropriate pixels.
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    // * Finally, handle all of the paths that still haven't terminated.
    //   (Easy way is to make them black or background-colored.)

    // TODO: perform one iteration of path tracing
	float tanx = std::tan(cam.fov.x*PI/180);
	float tany = std::tan(cam.fov.y*PI/180);
	glm::vec3 right = glm::cross(cam.view, cam.up);
	glm::vec3 perup = glm::cross(right, cam.view);

	rayBuilder << <blocksPerGrid, blockSize >> >(cam, dev_rays, tanx, tany, right, perup, pixelcount, iter);
	checkCUDAError("rayBuilder");
	//rayDebug<<<blocksPerGrid,blockSize>>>(cam,dev_image,dev_rays);

	int compacted = 0;
	dim3 blockSize1d(64, 1);
	dim3 blocksPerGrid1d((pixelcount - compacted + blockSize1d.x - 1) / blockSize1d.x, 1);
	int debug = 4;
	for (int i = 0; i < traceDepth; i++) {
		dim3 blockSize1d(64, 1);
		dim3 blocksPerGrid1d((pixelcount - compacted + blockSize1d.x - 1) / blockSize1d.x, 1);
		slingRays<<<blocksPerGrid1d, blockSize1d>>>(dev_rays, dev_geo, hst_scene->geoms.size(), dev_mat, dev_image, pixelcount, i, iter);
		checkCUDAError("Loop Fuck");
		//insert a streamcompact here
	}
	consumeRays << <blocksPerGrid1d, blockSize1d >> >(dev_rays, dev_image, pixelcount);
	checkCUDAError("Fuck");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}