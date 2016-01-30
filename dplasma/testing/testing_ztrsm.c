/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_solution( dague_context_t *dague, int loud,
                           int M, int N,
                           tiled_matrix_desc_t *ddescC,
                           tiled_matrix_desc_t *ddescCfinal );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int Aseed = 3872;
    int Cseed = 2873;
    dague_complex64_t alpha = 0.98;
    tiled_matrix_desc_t *ddescA;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(DAGUE_HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    /* initializing matrix structure */
    int Am = dplasma_imax(M, N);
    LDA = dplasma_imax(LDA, Am);
    LDC = dplasma_imax(LDC, M);
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, Am, 0, 0,
                               Am, Am, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC0, check,
        two_dim_block_cyclic, (&ddescC0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    /* Generate matrix A with diagonal dominance to keep stability during computation */
    dplasma_zplrnt( dague, 1, (tiled_matrix_desc_t *)&ddescA0, Aseed);
    /* Scale down the full matrix to keep stability in diag = PlasmaUnit case */
    dplasma_zlascal( dague, PlasmaUpperLower,
                     1. / (dague_complex64_t)Am,
                     (tiled_matrix_desc_t *)&ddescA0 );
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC, Cseed);
    if (check)
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescC, (tiled_matrix_desc_t *)&ddescC0 );
    if(loud > 2) printf("Done\n");

    if(!check)
    {
        PLASMA_enum side  = PlasmaLeft;
        PLASMA_enum uplo  = PlasmaLower;
        PLASMA_enum trans = PlasmaNoTrans;
        PLASMA_enum diag  = PlasmaUnit;

        /* Make A square */
        if (side == PlasmaLeft) {
            ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, M, M );
        } else {
            ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, N, N );
        }

        /* Compute b = 1/alpha * A * x */
        dplasma_ztrmm(dague, side, uplo, trans, diag, 1. / alpha,
                      ddescA, (tiled_matrix_desc_t *)&ddescC);

        PASTE_CODE_FLOPS(FLOPS_ZTRSM, (side, (DagDouble_t)M, (DagDouble_t)N));

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, ztrsm,
                                  (side, uplo, trans, diag, alpha,
                                   ddescA, (tiled_matrix_desc_t *)&ddescC));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, ztrsm);

        dplasma_ztrsm_Destruct( DAGUE_ztrsm );
        free(ddescA);
    }
    else
    {
        int s, u, t, d;
        int info_solution;

        for (s=0; s<2; s++) {
            /* Make A square */
            if (side[s] == PlasmaLeft) {
                ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, M, M );
            } else {
                ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, N, N );
            }

            for (u=0; u<2; u++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
                for (t=0; t<3; t++) {
#else
                for (t=0; t<2; t++) {
#endif
                    for (d=0; d<2; d++) {

                        if ( rank == 0 ) {
                            printf("***************************************************\n");
                            printf(" ----- TESTING ZTRSM (%s, %s, %s, %s) -------- \n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                        }

                        /* matrix generation */
                        printf("Generate matrices ... ");
                        dplasma_zlacpy( dague, PlasmaUpperLower,
                                        (tiled_matrix_desc_t *)&ddescC0,
                                        (tiled_matrix_desc_t *)&ddescC );
                        dplasma_ztrmm(dague, side[s], uplo[u], trans[t], diag[d], 1./alpha,
                                      ddescA, (tiled_matrix_desc_t *)&ddescC);
                        printf("Done\n");

                        /* Compute */
                        printf("Compute ... ... ");
                        dplasma_ztrsm(dague, side[s], uplo[u], trans[t], diag[d], alpha,
                                      ddescA, (tiled_matrix_desc_t *)&ddescC);
                        printf("Done\n");

                        /* Check the solution */
                        info_solution = check_solution(dague, rank == 0 ? loud : 0, M, N,
                                                       (tiled_matrix_desc_t*)&ddescC0,
                                                       (tiled_matrix_desc_t*)&ddescC);
                        if ( rank == 0 ) {
                            if (info_solution == 0) {
                                printf(" ---- TESTING ZTRSM (%s, %s, %s, %s) ...... PASSED !\n",
                                       sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                            }
                            else {
                                printf(" ---- TESTING ZTRSM (%s, %s, %s, %s) ... FAILED !\n",
                                       sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                                ret |= 1;
                            }
                            printf("***************************************************\n");
                        }
                    }
                }
#ifdef __UNUSED__
                }
#endif
            }
            free(ddescA);
        }
        dague_data_free(ddescC0.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC0);
    }

    dague_data_free(ddescA0.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
    dague_data_free(ddescC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);

    cleanup_dague(dague, iparam);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( dague_context_t *dague, int loud,
                           int M, int N,
                           tiled_matrix_desc_t *ddescC,
                           tiled_matrix_desc_t *ddescCfinal )
{
    int info_solution = 1;
    double Cinitnorm, Cdplasmanorm, Rnorm;
    double eps, result;

    eps = LAPACKE_dlamch_work('e');

    Cinitnorm    = dplasma_zlange( dague, PlasmaInfNorm, ddescC );
    Cdplasmanorm = dplasma_zlange( dague, PlasmaInfNorm, ddescCfinal );

    dplasma_zgeadd( dague, PlasmaNoTrans,
                    -1.0, (tiled_matrix_desc_t*)ddescC,
                     1.0, (tiled_matrix_desc_t*)ddescCfinal );

    Rnorm = dplasma_zlange( dague, PlasmaMaxNorm, ddescCfinal );

    result = Rnorm / (Cinitnorm * eps * dplasma_imax(M, N));

    if ( loud > 2 ) {
        printf("  ||x||_inf = %e, ||dplasma(A^(-1) b||_inf = %e, ||R||_m = %e, res = %e\n",
               Cinitnorm, Cdplasmanorm, Rnorm, result);
    }

    if (  isinf(Cdplasmanorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }

    return info_solution;
}
