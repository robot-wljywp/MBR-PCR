#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>

namespace py = pybind11;
using namespace std;

class PointXYZ
{
public:
    vector<float> pc;
    int numCount;
    float meanX;
    float meanY;
    float meanZ;
    float covX;
    float covY;
    float covZ;
    float covSumX;
    float covSumY;
    float covSumZ;


    PointXYZ() {
        meanX = 0;
        meanY = 0;
        meanZ = 0;
        covSumX = 0;
        covSumY = 0;
        covSumZ = 0;
        covX = 0;
        covY = 0;
        covZ = 0;
    }

    void add(float x,float y,float z)
    {
        pc.push_back(x);
        pc.push_back(y);
        pc.push_back(z);
    }

    int calDistribution()
    {
        numCount = pc.size() / 3;
        return numCount;
    }

    void Gaussian()
    {
        int pcSize = pc.size();
        for (int i = 0; i < pcSize; i+=3)
        {
            meanX += pc[i];
            meanY += pc[i + 1];
            meanZ += pc[i + 2];
        }

        meanX /= numCount;
        meanY /= numCount;
        meanZ /= numCount;

        for (int i = 0; i < pcSize; i+=3)
        {
            covSumX += pow(pc[i] - meanX, 2);
            covSumY += pow(pc[i + 1] - meanY, 2);
            covSumZ += pow(pc[i + 2] - meanZ, 2);
        }

        covX = covSumX / numCount;
        covY = covSumY / numCount;
        covZ = covSumZ / numCount;
    }
};

pair<py::array, py::array> distributionSample(py::array_t<float>& pc1, py::array_t<float>& pc2, py::array_t<float>& pcMin, py::array_t<float>& pcMax, float sampleDl)
{
    py::buffer_info reqPc1 = pc1.request();
    float *ptrPc1 = static_cast<float *>(reqPc1.ptr);
    py::buffer_info reqPc2 = pc2.request();
    float *ptrPc2 = static_cast<float *>(reqPc2.ptr);
    py::buffer_info req2 = pcMin.request();
    float *ptr2 = static_cast<float *>(req2.ptr);
    py::buffer_info req3 = pcMax.request();
    float *ptr3 = static_cast<float *>(req3.ptr);

    PointXYZ tmp1, tmp2;
    vector<float> retPc1, retPc2;
	unordered_map<size_t, PointXYZ> pcData1, pcData2;
    size_t mapIdx, intersectionSum, intersectionThres;
    size_t h1 = reqPc1.shape[0];
    size_t h2 = reqPc2.shape[0];
    size_t xIndex, yIndex, zIndex;
    float Gaussiankernel = 0;
    float GaussiankernalThres = 0;
    float GaussiankernelSum = 0;
    float xMin = *(ptr2 + 0);
    float xMax = *(ptr3 + 0);
    float yMin = *(ptr2 + 1);
    float yMax = *(ptr3 + 1);
    float zMin = *(ptr2 + 2);
    float zMax = *(ptr3 + 2);
	size_t sampleNX = (size_t)floor((xMax - xMin) / sampleDl) + 1;
	size_t sampleNY = (size_t)floor((yMax - yMin) / sampleDl) + 1;
    float tmpX, tmpY, tmpZ;

    for (int i = 0; i < h1; ++i)
    {
        tmpX = *(ptrPc1 + i * 3    );
        tmpY = *(ptrPc1 + i * 3 + 1);
        tmpZ = *(ptrPc1 + i * 3 + 2);

        xIndex = (size_t)floor((tmpX - xMin) / sampleDl);
        yIndex = (size_t)floor((tmpY - yMin) / sampleDl);
        zIndex = (size_t)floor((tmpZ - zMin) / sampleDl);
        mapIdx = xIndex + sampleNX * yIndex + sampleNX * sampleNY * zIndex;
        if(pcData1.count(mapIdx) < 1)
            pcData1.emplace(mapIdx, PointXYZ());

        pcData1[mapIdx].add(tmpX, tmpY, tmpZ);
    }
    for (int i = 0; i < h2; ++i)
    {
        tmpX = *(ptrPc2 + i * 3    );
        tmpY = *(ptrPc2 + i * 3 + 1);
        tmpZ = *(ptrPc2 + i * 3 + 2);

        xIndex = (size_t)floor((tmpX - xMin) / sampleDl);
        yIndex = (size_t)floor((tmpY - yMin) / sampleDl);
        zIndex = (size_t)floor((tmpZ - zMin) / sampleDl);
        mapIdx = xIndex + sampleNX * yIndex + sampleNX * sampleNY * zIndex;
        if(pcData2.count(mapIdx) < 1)
            pcData2.emplace(mapIdx, PointXYZ());

        pcData2[mapIdx].add(tmpX, tmpY, tmpZ);
    }

    std::set<int> keys1;
    for (auto it = pcData1.begin(); it != pcData1.end(); it++)
    {
        keys1.insert(it->first);
    }
    std::set<int> keys2;
    for (auto it = pcData2.begin(); it != pcData2.end(); it++)
    {
        keys2.insert(it->first);
    }

	std::set<int> intersection;
	std::set_intersection(keys1.begin(), keys1.end(), keys2.begin(), keys2.end(), std::inserter(intersection, intersection.begin()));

    intersectionSum = 0;

    std::vector<float> GaussiankernelVector(intersection.size());

    int j = 0;
    for (int i : intersection) {
        intersectionSum += abs(pcData1[i].calDistribution() - pcData2[i].calDistribution());

        pcData1[i].Gaussian();
        pcData2[i].Gaussian();

        Gaussiankernel = exp(- (pow(pcData1[i].meanX - pcData2[i].meanX, 2) / (4 * (pcData1[i].covX + pcData2[i].covX)) +
                                pow(pcData1[i].meanY - pcData2[i].meanY, 2) / (4 * (pcData1[i].covY + pcData2[i].covY)) +
                                pow(pcData1[i].meanZ - pcData2[i].meanZ, 2) / (4 * (pcData1[i].covZ + pcData2[i].covZ))));
        GaussiankernelVector[j] = Gaussiankernel;
        GaussiankernelSum += Gaussiankernel;
        ++j;
    }

    intersectionThres = int(intersectionSum / intersection.size()); 

    GaussiankernalThres = GaussiankernelSum / intersection.size();

	j = 0;
	for (int i : intersection) {
        if (abs(pcData1[i].calDistribution() - pcData2[i].calDistribution()) < intersectionThres
            && GaussiankernelVector[j] >= GaussiankernalThres
            )
        {
            retPc1.insert(retPc1.end(), pcData1[i].pc.begin(), pcData1[i].pc.end());
            retPc2.insert(retPc2.end(), pcData2[i].pc.begin(), pcData2[i].pc.end());
        }
        ++j;
	}

    size_t ndim = 2;
    vector<size_t> shape1   = { retPc1.size() / 3 , 3 };
    vector<size_t> shape2   = { retPc2.size() / 3 , 3 };
    vector<size_t> strides = { sizeof(float)*3 , sizeof(float) };

    return
    make_pair(
    py::array(py::buffer_info(
        retPc1.data(),                           /* data as contiguous array  */
        sizeof(float),                          /* size of one scalar        */
        py::format_descriptor<float>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape1,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    )),
        py::array(py::buffer_info(
        retPc2.data(),                           /* data as contiguous array  */
        sizeof(float),                          /* size of one scalar        */
        py::format_descriptor<float>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape2,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    )));
}

PYBIND11_MODULE(distributionSample, m) {

    m.doc() = "Sample point cloud by distribution";
    m.def("distributionSample", &distributionSample);

}