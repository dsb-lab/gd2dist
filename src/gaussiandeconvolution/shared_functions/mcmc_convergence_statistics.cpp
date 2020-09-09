#include <vector>
#include <cmath>

#include "mcmc_convergence_statistics.h"

double rstat(std::vector<double>& vphi, int nChains){
    int totalLength = vphi.size(); 
    //Split each chain in 2
    int n = int(totalLength/nChains/2);
    //Number of segments
    int m = 2*nChains;
    //Auxiliar values
    std::vector<double> phij(m,0);
    double phi = 0;
    std::vector<double> sj(m,0);
    double B = 0;
    double W = 0;

    //Compute phij
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            phij[i] += vphi[i*n+j];
        }
        phij[i] /= n;
    }
    //Compute phi
    for (int i = 0; i < m; i++){
        phi += phij[i];
    }
    phi /= m;
    //Compute B
    for (int i = 0; i < m; i++){
        B += std::pow(phij[i]-phi,2);
    }
    B *= n/(m-1);
    //Compute sj
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            sj[i] += std::pow(vphi[i*n+j]-phij[i],2);
        }
        sj[i] /= (n-1);
    }
    //Compute W
    for (int i = 0; i < m; i++){
        W += sj[i];
    }
    W /= m;

    if(W == 0){
        return 1;
    }
    else{
        return std::sqrt(1-1/n+B/(W*n));
    }
}

double rstat(std::vector<std::vector<double>>& vphi){
    std::vector<double> vphi2(vphi[0]);

    for(int i = 1; i < vphi.size(); i++){
        vphi2.insert( vphi2.end(), vphi[i].begin(), vphi[i].end() );
    }

    return rstat(vphi2, vphi.size());
}

double effnumber(std::vector<double>& vphi, int nChains){
    int totalLength = vphi.size(); 
    //Split each chain in 2
    int n = int(totalLength/nChains/2);
    //Number of segments
    int m = 2*nChains;
    //Auxiliar values
    std::vector<double> phij(m,0);
    double phi = 0;
    std::vector<double> sj(m,0);
    double B = 0;
    double W = 0;
    double Var = 0;
    double rhoT = 0;
    double rho0 = 0;
    double rho1 = 0;
    int t = 1;
    double Vt = 0;

    //Compute phij
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            phij[i] += vphi[i*n+j];
        }
        phij[i] /= n;
    }
    //Compute phi
    for (int i = 0; i < m; i++){
        phi += phij[i];
    }
    phi /= m;
    //Compute B
    for (int i = 0; i < m; i++){
        B += std::pow(phij[i]-phi,2);
    }
    B *= n/(m-1);
    //Compute sj
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            sj[i] += std::pow(vphi[i*n+j]-phij[i],2);
        }
        sj[i] /= (n-1);
    }
    //Compute W
    for (int i = 0; i < m; i++){
        W += sj[i];
    }
    W /= m;
    //Compute Var+
    Var = (n-1)/n*W+B/n;

    //Compute neff
    do{
        Vt = 0;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n-t; j++){
                Vt += std::pow(vphi[n*i+j]-vphi[n*i+j+t],2);
            }
        }
        Vt /= m*(n-t);
        rho1 = rho0;
        rho0 = 1-Vt/(2*Var);
        if (rho1+rho0 >= 0){
            rhoT += rho0;
        }
        else{
            break;
        }

        t += 1;
    }while(t < 2*n);

    return m*n/(1+2*rhoT);
}

double effnumber(std::vector<std::vector<double>>& vphi){
    std::vector<double> vphi2(vphi[0]);

    for(int i = 1; i < vphi.size(); i++){
        vphi2.insert( vphi2.end(), vphi[i].begin(), vphi[i].end() );
    }

    return effnumber(vphi2, vphi.size());
}
