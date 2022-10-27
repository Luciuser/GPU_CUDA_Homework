#include "cmesh.h"
#include <set>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "tmbvh.hpp"

#define MAX_CD_PAIRS 4096*1024

extern mesh *cloths[16];
extern mesh *lions[16];

extern vector<int> vtx_set;
extern set<int> cloth_set;
extern set<int> lion_set;
static bvh *bvhCloth = NULL;
static bvh *bvhBody = NULL;

bool findd;

#include <omp.h>

# define	TIMING_BEGIN \
	{double tmp_timing_start = omp_get_wtime();

# define	TIMING_END(message) \
	{double tmp_timing_finish = omp_get_wtime();\
	double  tmp_timing_duration = tmp_timing_finish - tmp_timing_start;\
	printf("%s: %2.5f seconds\n", (message), tmp_timing_duration);}}


// CPU with BVH
void buildBVH()
{

	TIMING_BEGIN
	static std::vector<mesh *> meshes;

		if (bvhCloth == NULL)
		{
			for (int i = 0; i < 16; i++)
				if (cloths[i] != NULL)
					meshes.push_back(cloths[i]);

			bvhCloth = new bvh(meshes);
		}

	bvhCloth->refit(meshes);
	TIMING_END("bvh done...")
}

void drawBVH(int level)
{
	if (bvhCloth == NULL) return;
	bvhCloth->visualize(level);
}


int maxLevel = 60;
std::vector<std::vector<int>> gAdjInfo; // �洢����ÿ�������ε��ڽ������Σ��ڽ�ָ�������������εİ�Χ���ཻ��
std::vector<REAL> gIntensity[2];
int currentPass = -1; // ���������ʾ��ǰ gIntensity �ǵڼ������� 0 �� 1 ֮�佻��
std::vector<int> gSources; // ��Դ��

void doPropogate()
{
	int prevPass = currentPass;
	currentPass = 1 - currentPass;

	mesh *mc = cloths[0];
	int num = mc->getNbFaces();

	TIMING_BEGIN
		for (int i = 0; i < num; i++) {
			std::vector<int> &adjs = gAdjInfo[i];
			// printf("adjs size is: %d", adjs.size());
			gIntensity[currentPass][i] = gIntensity[prevPass][i];
			for (int j = 0; j < adjs.size(); j++) {
				int tj = adjs[j];
				gIntensity[currentPass][i] += gIntensity[prevPass][tj];
			}

			gIntensity[currentPass][i] /= REAL(adjs.size() + 1);
		}

	for (int i = 0; i < gSources.size(); i++) {
		gIntensity[currentPass][gSources[i]] = 1.0;
	}
	TIMING_END("propogating...")
}

extern void buildIt();
extern "C" int doPropogateGPU();

void doIt()
{
	// ������ update() ������ÿ֡����һ��
	if (currentPass == -1) {
		// ��һ��ִ��
		currentPass = 0;
		buildIt();

		gSources.push_back(100);
		gSources.push_back(10);
		gSources.push_back(200);

		mesh *mc = cloths[0];
		int num = mc->getNbFaces();
		gIntensity[0].resize(num);
		gIntensity[1].resize(num);

		for (int i = 0; i < num; i++) {
			gIntensity[currentPass][i] = 0;
		}

		for (int i = 0; i < gSources.size(); i++) {
			gIntensity[currentPass][gSources[i]] = 1.0;
		}
	}
	else {
		//doPropogate();
		doPropogateGPU();
	}
}

void buildIt()
{
	mesh *mc = cloths[0];
	int num = mc->getNbFaces();
	gAdjInfo.clear();
	gAdjInfo.resize(num);

	TIMING_BEGIN
	for (int i = 0; i < num; i++) {
		BOX bx = mc->getTriBox(i);
		bvhCloth->query(bx, gAdjInfo[i], i);
	}
	TIMING_END("build adj info")
}
