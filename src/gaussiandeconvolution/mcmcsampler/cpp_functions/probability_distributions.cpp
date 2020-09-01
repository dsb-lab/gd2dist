#include <math.h>
#include <random>

double
gaussian_pdf(double x, double mu, double sigma){
    return  - std::pow( x - mu , 2 ) / ( 2 * std::pow( sigma , 2 ) )  - std::log( std::sqrt( 2 * M_PI) * sigma );
}

void
multinomial_1(std::mt19937 &r, std::vector<double> & p, std::vector<int> & x){
    double cum = 0;
    double tot = 0;
    int pos = 0;
    int l = p.size();
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    //Normalize
    for ( int i = 0 ; i < l ; i++ ){
        tot += p[i];
    }
    //Sample
    double v = tot * distribution(r);
    for ( int i = 0 ; i < l ; i++){
        cum += p[i];
        if ( cum > v ){
            x[i] = 1;
            pos = i;
            break;
        }else{
            x[i] = 0;
        }
    }
    for ( int i = pos + 1 ; i < l ; i++){
        x[i] = 0;
    }

    return;
}

void
dirichlet(std::mt19937 & r, std::vector<double> & a, std::vector<double> & x){
    int l = a.size();
    double tot = 0;

    //Sample gamma
    for ( int i = 0; i < l; i++ ){
        std::gamma_distribution<double> gamma(a[i],1);
        x[i] = gamma(r);
        tot += x[i];
    }
    //Normalize
    for ( int i = 0; i < l; i++ ){
        x[i] /= tot;
    }

    return;
}
