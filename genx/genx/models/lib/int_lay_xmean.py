'''A library for the analytically calculated reflection matrices from a sample
 as modelled in the mag_refl model. The code has been automatically 
generated from MuPad notebook calculations by the script mupad2python.py.
Programmer: Matts Bjorck 
Generated 2014-05-06 19:55:13.436879'''

import numpy as np

def calc_xrmr_Xmean(lamda, X_l, X_lu, X_u, u, u_l, u_u, dd_u, dd_l, sigma, sigma_l, sigma_u):
    '''Function to assemble Xmean for the xrmr module.'''

    kappa=2*np.pi/lamda
    Xmean=np.empty(X_l.shape, dtype=np.complex128)

    t1=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[3, :-1]-u_l[3, :-1])**2
    t2=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[3, :-1]-u_l[2, :-1])**2
    t3=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[2, :-1]-u_l[3, :-1])**2
    t4=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[3, :-1]-u_l[1, :-1])**2
    t5=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[2, :-1]-u_l[2, :-1])**2
    t6=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[1, :-1]-u_l[3, :-1])**2
    t7=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[3, :-1]-u_l[0, :-1])**2
    t8=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[2, :-1]-u_l[1, :-1])**2
    t9=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[1, :-1]-u_l[2, :-1])**2
    t10=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[0, :-1]-u_l[3, :-1])**2
    t11=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[2, :-1]-u_l[0, :-1])**2
    t12=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[1, :-1]-u_l[1, :-1])**2
    t13=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[0, :-1]-u_l[2, :-1])**2
    t14=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[1, :-1]-u_l[0, :-1])**2
    t15=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[0, :-1]-u_l[1, :-1])**2
    t16=(1./2.)*kappa**2*sigma_l[:-1]**2*(u[0, :-1]-u_l[0, :-1])**2
    t17=(1./2.)*kappa**2*sigma[1:]**2*(u[3, 1:]-u[3, :-1])**2
    t18=(1./2.)*kappa**2*sigma[1:]**2*(u[3, 1:]-u[2, :-1])**2
    t19=(1./2.)*kappa**2*sigma[1:]**2*(u[2, 1:]-u[3, :-1])**2
    t20=(1./2.)*kappa**2*sigma[1:]**2*(u[3, 1:]-u[1, :-1])**2
    t21=(1./2.)*kappa**2*sigma[1:]**2*(u[2, 1:]-u[2, :-1])**2
    t22=(1./2.)*kappa**2*sigma[1:]**2*(u[1, 1:]-u[3, :-1])**2
    t23=(1./2.)*kappa**2*sigma[1:]**2*(u[3, 1:]-u[0, :-1])**2
    t24=(1./2.)*kappa**2*sigma[1:]**2*(u[2, 1:]-u[1, :-1])**2
    t25=(1./2.)*kappa**2*sigma[1:]**2*(u[1, 1:]-u[2, :-1])**2
    t26=(1./2.)*kappa**2*sigma[1:]**2*(u[0, 1:]-u[3, :-1])**2
    t27=(1./2.)*kappa**2*sigma[1:]**2*(u[2, 1:]-u[0, :-1])**2
    t28=(1./2.)*kappa**2*sigma[1:]**2*(u[1, 1:]-u[1, :-1])**2
    t29=(1./2.)*kappa**2*sigma[1:]**2*(u[0, 1:]-u[2, :-1])**2
    t30=(1./2.)*kappa**2*sigma[1:]**2*(u[1, 1:]-u[0, :-1])**2
    t31=(1./2.)*kappa**2*sigma[1:]**2*(u[0, 1:]-u[1, :-1])**2
    t32=(1./2.)*kappa**2*sigma[1:]**2*(u[0, 1:]-u[0, :-1])**2
    t33=(1./2.)*kappa**2*sigma_u[1:]**2*(u[3, 1:]-u_u[3, 1:])**2
    t34=(1./2.)*kappa**2*sigma_u[1:]**2*(u[3, 1:]-u_u[2, 1:])**2
    t35=(1./2.)*kappa**2*sigma_u[1:]**2*(u[2, 1:]-u_u[3, 1:])**2
    t36=(1./2.)*kappa**2*sigma_u[1:]**2*(u[3, 1:]-u_u[1, 1:])**2
    t37=(1./2.)*kappa**2*sigma_u[1:]**2*(u[2, 1:]-u_u[2, 1:])**2
    t38=(1./2.)*kappa**2*sigma_u[1:]**2*(u[1, 1:]-u_u[3, 1:])**2
    t39=(1./2.)*kappa**2*sigma_u[1:]**2*(u[3, 1:]-u_u[0, 1:])**2
    t40=(1./2.)*kappa**2*sigma_u[1:]**2*(u[2, 1:]-u_u[1, 1:])**2
    t41=(1./2.)*kappa**2*sigma_u[1:]**2*(u[1, 1:]-u_u[2, 1:])**2
    t42=(1./2.)*kappa**2*sigma_u[1:]**2*(u[0, 1:]-u_u[3, 1:])**2
    t43=(1./2.)*kappa**2*sigma_u[1:]**2*(u[2, 1:]-u_u[0, 1:])**2
    t44=(1./2.)*kappa**2*sigma_u[1:]**2*(u[1, 1:]-u_u[1, 1:])**2
    t45=(1./2.)*kappa**2*sigma_u[1:]**2*(u[0, 1:]-u_u[2, 1:])**2
    t46=(1./2.)*kappa**2*sigma_u[1:]**2*(u[1, 1:]-u_u[0, 1:])**2
    t47=(1./2.)*kappa**2*sigma_u[1:]**2*(u[0, 1:]-u_u[1, 1:])**2
    t48=(1./2.)*kappa**2*sigma_u[1:]**2*(u[0, 1:]-u_u[0, 1:])**2
    t49=kappa*(dd_u[1:]*u_u[3, 1:]*1.0J+dd_l[:-1]*u_l[3, :-1]*1.0J)
    t50=kappa*(dd_u[1:]*u_u[3, 1:]*1.0J+dd_l[:-1]*u_l[2, :-1]*1.0J)
    t51=kappa*(dd_u[1:]*u_u[2, 1:]*1.0J+dd_l[:-1]*u_l[3, :-1]*1.0J)
    t52=kappa*(dd_u[1:]*u_u[3, 1:]*1.0J+dd_l[:-1]*u_l[1, :-1]*1.0J)
    t53=kappa*(dd_u[1:]*u_u[2, 1:]*1.0J+dd_l[:-1]*u_l[2, :-1]*1.0J)
    t54=kappa*(dd_u[1:]*u_u[1, 1:]*1.0J+dd_l[:-1]*u_l[3, :-1]*1.0J)
    t55=kappa*(dd_u[1:]*u_u[3, 1:]*1.0J+dd_l[:-1]*u_l[0, :-1]*1.0J)
    t56=kappa*(dd_u[1:]*u_u[2, 1:]*1.0J+dd_l[:-1]*u_l[1, :-1]*1.0J)
    t57=kappa*(dd_u[1:]*u_u[1, 1:]*1.0J+dd_l[:-1]*u_l[2, :-1]*1.0J)
    t58=kappa*(dd_u[1:]*u_u[0, 1:]*1.0J+dd_l[:-1]*u_l[3, :-1]*1.0J)
    t59=kappa*(dd_u[1:]*u_u[2, 1:]*1.0J+dd_l[:-1]*u_l[0, :-1]*1.0J)
    t60=kappa*(dd_u[1:]*u_u[1, 1:]*1.0J+dd_l[:-1]*u_l[1, :-1]*1.0J)
    t61=kappa*(dd_u[1:]*u_u[0, 1:]*1.0J+dd_l[:-1]*u_l[2, :-1]*1.0J)
    t62=kappa*(dd_u[1:]*u_u[1, 1:]*1.0J+dd_l[:-1]*u_l[0, :-1]*1.0J)
    t63=kappa*(dd_u[1:]*u_u[0, 1:]*1.0J+dd_l[:-1]*u_l[1, :-1]*1.0J)
    t64=kappa*(dd_u[1:]*u_u[0, 1:]*1.0J+dd_l[:-1]*u_l[0, :-1]*1.0J)

    Xmean[0, 0]=np.exp(- t16-t32-t48-t64)*X_u[0, 0]*X_lu[0, 0]*X_l[0, 0]+np.exp(- t16-t32-t47-t62)*X_u[1, 0]*X_lu[0, 1]* \
                X_l[0, 0]+np.exp(- t15-t32-t48-t63)*X_u[0, 0]*X_lu[1, 0]*X_l[0, 1]+np.exp(- t16-t32-t45-t59)*X_u[2, 0]* \
                X_lu[0, 2]*X_l[0, 0]+np.exp(- t13-t32-t48-t61)*X_u[0, 0]*X_lu[2, 0]*X_l[0, 2]+np.exp(- t15-t32-t47-t60)* \
                X_u[1, 0]*X_lu[1, 1]*X_l[0, 1]+np.exp(- t16-t32-t42-t55)*X_u[3, 0]*X_lu[0, 3]*X_l[0, 0]+np.exp(
        - t10-t32-t48-t58)*X_u[0, 0]*X_lu[3, 0]*X_l[0, 3]+np.exp(- t15-t32-t45-t56)*X_u[2, 0]*X_lu[1, 2]*X_l[
                    0, 1]+np.exp(- t13-t32-t47-t57)*X_u[1, 0]*X_lu[2, 1]*X_l[0, 2]+np.exp(- t15-t32-t42-t52)*X_u[3, 0]* \
                X_lu[1, 3]*X_l[0, 1]+np.exp(- t10-t32-t47-t54)*X_u[1, 0]*X_lu[3, 1]*X_l[0, 3]+np.exp(- t13-t32-t45-t53)* \
                X_u[2, 0]*X_lu[2, 2]*X_l[0, 2]+np.exp(- t13-t32-t42-t50)*X_u[3, 0]*X_lu[2, 3]*X_l[0, 2]+np.exp(
        - t10-t32-t45-t51)*X_u[2, 0]*X_lu[3, 2]*X_l[0, 3]+np.exp(- t10-t32-t42-t49)*X_u[3, 0]*X_lu[3, 3]*X_l[0, 3]
    Xmean[0, 1]=np.exp(- t16-t30-t46-t64)*X_u[0, 1]*X_lu[0, 0]*X_l[0, 0]+np.exp(- t16-t30-t44-t62)*X_u[1, 1]*X_lu[0, 1]* \
                X_l[0, 0]+np.exp(- t15-t30-t46-t63)*X_u[0, 1]*X_lu[1, 0]*X_l[0, 1]+np.exp(- t16-t30-t41-t59)*X_u[2, 1]* \
                X_lu[0, 2]*X_l[0, 0]+np.exp(- t15-t30-t44-t60)*X_u[1, 1]*X_lu[1, 1]*X_l[0, 1]+np.exp(- t16-t30-t38-t55)* \
                X_u[3, 1]*X_lu[0, 3]*X_l[0, 0]+np.exp(- t13-t30-t46-t61)*X_u[0, 1]*X_lu[2, 0]*X_l[0, 2]+np.exp(
        - t15-t30-t41-t56)*X_u[2, 1]*X_lu[1, 2]*X_l[0, 1]+np.exp(- t10-t30-t46-t58)*X_u[0, 1]*X_lu[3, 0]*X_l[
                    0, 3]+np.exp(- t13-t30-t44-t57)*X_u[1, 1]*X_lu[2, 1]*X_l[0, 2]+np.exp(- t15-t30-t38-t52)*X_u[3, 1]* \
                X_lu[1, 3]*X_l[0, 1]+np.exp(- t13-t30-t41-t53)*X_u[2, 1]*X_lu[2, 2]*X_l[0, 2]+np.exp(- t10-t30-t44-t54)* \
                X_u[1, 1]*X_lu[3, 1]*X_l[0, 3]+np.exp(- t13-t30-t38-t50)*X_u[3, 1]*X_lu[2, 3]*X_l[0, 2]+np.exp(
        - t10-t30-t41-t51)*X_u[2, 1]*X_lu[3, 2]*X_l[0, 3]+np.exp(- t10-t30-t38-t49)*X_u[3, 1]*X_lu[3, 3]*X_l[0, 3]
    Xmean[0, 2]=np.exp(- t16-t27-t43-t64)*X_u[0, 2]*X_lu[0, 0]*X_l[0, 0]+np.exp(- t16-t27-t40-t62)*X_u[1, 2]*X_lu[0, 1]* \
                X_l[0, 0]+np.exp(- t15-t27-t43-t63)*X_u[0, 2]*X_lu[1, 0]*X_l[0, 1]+np.exp(- t16-t27-t37-t59)*X_u[2, 2]* \
                X_lu[0, 2]*X_l[0, 0]+np.exp(- t15-t27-t40-t60)*X_u[1, 2]*X_lu[1, 1]*X_l[0, 1]+np.exp(- t16-t27-t35-t55)* \
                X_u[3, 2]*X_lu[0, 3]*X_l[0, 0]+np.exp(- t13-t27-t43-t61)*X_u[0, 2]*X_lu[2, 0]*X_l[0, 2]+np.exp(
        - t15-t27-t37-t56)*X_u[2, 2]*X_lu[1, 2]*X_l[0, 1]+np.exp(- t13-t27-t40-t57)*X_u[1, 2]*X_lu[2, 1]*X_l[
                    0, 2]+np.exp(- t10-t27-t43-t58)*X_u[0, 2]*X_lu[3, 0]*X_l[0, 3]+np.exp(- t15-t27-t35-t52)*X_u[3, 2]* \
                X_lu[1, 3]*X_l[0, 1]+np.exp(- t13-t27-t37-t53)*X_u[2, 2]*X_lu[2, 2]*X_l[0, 2]+np.exp(- t10-t27-t40-t54)* \
                X_u[1, 2]*X_lu[3, 1]*X_l[0, 3]+np.exp(- t10-t27-t37-t51)*X_u[2, 2]*X_lu[3, 2]*X_l[0, 3]+np.exp(
        - t13-t27-t35-t50)*X_u[3, 2]*X_lu[2, 3]*X_l[0, 2]+np.exp(- t10-t27-t35-t49)*X_u[3, 2]*X_lu[3, 3]*X_l[0, 3]
    Xmean[0, 3]=np.exp(- t16-t23-t39-t64)*X_u[0, 3]*X_lu[0, 0]*X_l[0, 0]+np.exp(- t16-t23-t36-t62)*X_u[1, 3]*X_lu[0, 1]* \
                X_l[0, 0]+np.exp(- t15-t23-t39-t63)*X_u[0, 3]*X_lu[1, 0]*X_l[0, 1]+np.exp(- t16-t23-t34-t59)*X_u[2, 3]* \
                X_lu[0, 2]*X_l[0, 0]+np.exp(- t15-t23-t36-t60)*X_u[1, 3]*X_lu[1, 1]*X_l[0, 1]+np.exp(- t13-t23-t39-t61)* \
                X_u[0, 3]*X_lu[2, 0]*X_l[0, 2]+np.exp(- t16-t23-t33-t55)*X_u[3, 3]*X_lu[0, 3]*X_l[0, 0]+np.exp(
        - t15-t23-t34-t56)*X_u[2, 3]*X_lu[1, 2]*X_l[0, 1]+np.exp(- t13-t23-t36-t57)*X_u[1, 3]*X_lu[2, 1]*X_l[
                    0, 2]+np.exp(- t10-t23-t39-t58)*X_u[0, 3]*X_lu[3, 0]*X_l[0, 3]+np.exp(- t10-t23-t36-t54)*X_u[1, 3]* \
                X_lu[3, 1]*X_l[0, 3]+np.exp(- t13-t23-t34-t53)*X_u[2, 3]*X_lu[2, 2]*X_l[0, 2]+np.exp(- t15-t23-t33-t52)* \
                X_u[3, 3]*X_lu[1, 3]*X_l[0, 1]+np.exp(- t10-t23-t34-t51)*X_u[2, 3]*X_lu[3, 2]*X_l[0, 3]+np.exp(
        - t13-t23-t33-t50)*X_u[3, 3]*X_lu[2, 3]*X_l[0, 2]+np.exp(- t10-t23-t33-t49)*X_u[3, 3]*X_lu[3, 3]*X_l[0, 3]
    Xmean[1, 0]=np.exp(- t14-t31-t48-t64)*X_u[0, 0]*X_lu[0, 0]*X_l[1, 0]+np.exp(- t12-t31-t48-t63)*X_u[0, 0]*X_lu[1, 0]* \
                X_l[1, 1]+np.exp(- t14-t31-t47-t62)*X_u[1, 0]*X_lu[0, 1]*X_l[1, 0]+np.exp(- t9-t31-t48-t61)*X_u[0, 0]* \
                X_lu[2, 0]*X_l[1, 2]+np.exp(- t14-t31-t45-t59)*X_u[2, 0]*X_lu[0, 2]*X_l[1, 0]+np.exp(- t12-t31-t47-t60)* \
                X_u[1, 0]*X_lu[1, 1]*X_l[1, 1]+np.exp(- t14-t31-t42-t55)*X_u[3, 0]*X_lu[0, 3]*X_l[1, 0]+np.exp(
        - t6-t31-t48-t58)*X_u[0, 0]*X_lu[3, 0]*X_l[1, 3]+np.exp(- t9-t31-t47-t57)*X_u[1, 0]*X_lu[2, 1]*X_l[1, 2]+np.exp(
        - t12-t31-t45-t56)*X_u[2, 0]*X_lu[1, 2]*X_l[1, 1]+np.exp(- t12-t31-t42-t52)*X_u[3, 0]*X_lu[1, 3]*X_l[
                    1, 1]+np.exp(- t6-t31-t47-t54)*X_u[1, 0]*X_lu[3, 1]*X_l[1, 3]+np.exp(- t9-t31-t45-t53)*X_u[2, 0]* \
                X_lu[2, 2]*X_l[1, 2]+np.exp(- t9-t31-t42-t50)*X_u[3, 0]*X_lu[2, 3]*X_l[1, 2]+np.exp(- t6-t31-t45-t51)* \
                X_u[2, 0]*X_lu[3, 2]*X_l[1, 3]+np.exp(- t6-t31-t42-t49)*X_u[3, 0]*X_lu[3, 3]*X_l[1, 3]
    Xmean[1, 1]=np.exp(- t14-t28-t46-t64)*X_u[0, 1]*X_lu[0, 0]*X_l[1, 0]+np.exp(- t14-t28-t44-t62)*X_u[1, 1]*X_lu[0, 1]* \
                X_l[1, 0]+np.exp(- t12-t28-t46-t63)*X_u[0, 1]*X_lu[1, 0]*X_l[1, 1]+np.exp(- t14-t28-t41-t59)*X_u[2, 1]* \
                X_lu[0, 2]*X_l[1, 0]+np.exp(- t9-t28-t46-t61)*X_u[0, 1]*X_lu[2, 0]*X_l[1, 2]+np.exp(- t12-t28-t44-t60)* \
                X_u[1, 1]*X_lu[1, 1]*X_l[1, 1]+np.exp(- t14-t28-t38-t55)*X_u[3, 1]*X_lu[0, 3]*X_l[1, 0]+np.exp(
        - t12-t28-t41-t56)*X_u[2, 1]*X_lu[1, 2]*X_l[1, 1]+np.exp(- t6-t28-t46-t58)*X_u[0, 1]*X_lu[3, 0]*X_l[
                    1, 3]+np.exp(- t9-t28-t44-t57)*X_u[1, 1]*X_lu[2, 1]*X_l[1, 2]+np.exp(- t12-t28-t38-t52)*X_u[3, 1]* \
                X_lu[1, 3]*X_l[1, 1]+np.exp(- t9-t28-t41-t53)*X_u[2, 1]*X_lu[2, 2]*X_l[1, 2]+np.exp(- t6-t28-t44-t54)* \
                X_u[1, 1]*X_lu[3, 1]*X_l[1, 3]+np.exp(- t9-t28-t38-t50)*X_u[3, 1]*X_lu[2, 3]*X_l[1, 2]+np.exp(
        - t6-t28-t41-t51)*X_u[2, 1]*X_lu[3, 2]*X_l[1, 3]+np.exp(- t6-t28-t38-t49)*X_u[3, 1]*X_lu[3, 3]*X_l[1, 3]
    Xmean[1, 2]=np.exp(- t14-t24-t43-t64)*X_u[0, 2]*X_lu[0, 0]*X_l[1, 0]+np.exp(- t14-t24-t40-t62)*X_u[1, 2]*X_lu[0, 1]* \
                X_l[1, 0]+np.exp(- t12-t24-t43-t63)*X_u[0, 2]*X_lu[1, 0]*X_l[1, 1]+np.exp(- t14-t24-t37-t59)*X_u[2, 2]* \
                X_lu[0, 2]*X_l[1, 0]+np.exp(- t12-t24-t40-t60)*X_u[1, 2]*X_lu[1, 1]*X_l[1, 1]+np.exp(- t9-t24-t43-t61)* \
                X_u[0, 2]*X_lu[2, 0]*X_l[1, 2]+np.exp(- t14-t24-t35-t55)*X_u[3, 2]*X_lu[0, 3]*X_l[1, 0]+np.exp(
        - t12-t24-t37-t56)*X_u[2, 2]*X_lu[1, 2]*X_l[1, 1]+np.exp(- t9-t24-t40-t57)*X_u[1, 2]*X_lu[2, 1]*X_l[
                    1, 2]+np.exp(- t6-t24-t43-t58)*X_u[0, 2]*X_lu[3, 0]*X_l[1, 3]+np.exp(- t9-t24-t37-t53)*X_u[2, 2]* \
                X_lu[2, 2]*X_l[1, 2]+np.exp(- t12-t24-t35-t52)*X_u[3, 2]*X_lu[1, 3]*X_l[1, 1]+np.exp(- t6-t24-t40-t54)* \
                X_u[1, 2]*X_lu[3, 1]*X_l[1, 3]+np.exp(- t6-t24-t37-t51)*X_u[2, 2]*X_lu[3, 2]*X_l[1, 3]+np.exp(
        - t9-t24-t35-t50)*X_u[3, 2]*X_lu[2, 3]*X_l[1, 2]+np.exp(- t6-t24-t35-t49)*X_u[3, 2]*X_lu[3, 3]*X_l[1, 3]
    Xmean[1, 3]=np.exp(- t14-t20-t39-t64)*X_u[0, 3]*X_lu[0, 0]*X_l[1, 0]+np.exp(- t14-t20-t36-t62)*X_u[1, 3]*X_lu[0, 1]* \
                X_l[1, 0]+np.exp(- t12-t20-t39-t63)*X_u[0, 3]*X_lu[1, 0]*X_l[1, 1]+np.exp(- t14-t20-t34-t59)*X_u[2, 3]* \
                X_lu[0, 2]*X_l[1, 0]+np.exp(- t12-t20-t36-t60)*X_u[1, 3]*X_lu[1, 1]*X_l[1, 1]+np.exp(- t9-t20-t39-t61)* \
                X_u[0, 3]*X_lu[2, 0]*X_l[1, 2]+np.exp(- t9-t20-t36-t57)*X_u[1, 3]*X_lu[2, 1]*X_l[1, 2]+np.exp(
        - t12-t20-t34-t56)*X_u[2, 3]*X_lu[1, 2]*X_l[1, 1]+np.exp(- t14-t20-t33-t55)*X_u[3, 3]*X_lu[0, 3]*X_l[
                    1, 0]+np.exp(- t6-t20-t39-t58)*X_u[0, 3]*X_lu[3, 0]*X_l[1, 3]+np.exp(- t6-t20-t36-t54)*X_u[1, 3]* \
                X_lu[3, 1]*X_l[1, 3]+np.exp(- t9-t20-t34-t53)*X_u[2, 3]*X_lu[2, 2]*X_l[1, 2]+np.exp(- t12-t20-t33-t52)* \
                X_u[3, 3]*X_lu[1, 3]*X_l[1, 1]+np.exp(- t6-t20-t34-t51)*X_u[2, 3]*X_lu[3, 2]*X_l[1, 3]+np.exp(
        - t9-t20-t33-t50)*X_u[3, 3]*X_lu[2, 3]*X_l[1, 2]+np.exp(- t6-t20-t33-t49)*X_u[3, 3]*X_lu[3, 3]*X_l[1, 3]
    Xmean[2, 0]=np.exp(- t11-t29-t48-t64)*X_u[0, 0]*X_lu[0, 0]*X_l[2, 0]+np.exp(- t8-t29-t48-t63)*X_u[0, 0]*X_lu[1, 0]* \
                X_l[2, 1]+np.exp(- t11-t29-t47-t62)*X_u[1, 0]*X_lu[0, 1]*X_l[2, 0]+np.exp(- t5-t29-t48-t61)*X_u[0, 0]* \
                X_lu[2, 0]*X_l[2, 2]+np.exp(- t8-t29-t47-t60)*X_u[1, 0]*X_lu[1, 1]*X_l[2, 1]+np.exp(- t11-t29-t45-t59)* \
                X_u[2, 0]*X_lu[0, 2]*X_l[2, 0]+np.exp(- t11-t29-t42-t55)*X_u[3, 0]*X_lu[0, 3]*X_l[2, 0]+np.exp(
        - t3-t29-t48-t58)*X_u[0, 0]*X_lu[3, 0]*X_l[2, 3]+np.exp(- t5-t29-t47-t57)*X_u[1, 0]*X_lu[2, 1]*X_l[2, 2]+np.exp(
        - t8-t29-t45-t56)*X_u[2, 0]*X_lu[1, 2]*X_l[2, 1]+np.exp(- t8-t29-t42-t52)*X_u[3, 0]*X_lu[1, 3]*X_l[2, 1]+np.exp(
        - t5-t29-t45-t53)*X_u[2, 0]*X_lu[2, 2]*X_l[2, 2]+np.exp(- t3-t29-t47-t54)*X_u[1, 0]*X_lu[3, 1]*X_l[2, 3]+np.exp(
        - t5-t29-t42-t50)*X_u[3, 0]*X_lu[2, 3]*X_l[2, 2]+np.exp(- t3-t29-t45-t51)*X_u[2, 0]*X_lu[3, 2]*X_l[2, 3]+np.exp(
        - t3-t29-t42-t49)*X_u[3, 0]*X_lu[3, 3]*X_l[2, 3]
    Xmean[2, 1]=np.exp(- t11-t25-t46-t64)*X_u[0, 1]*X_lu[0, 0]*X_l[2, 0]+np.exp(- t8-t25-t46-t63)*X_u[0, 1]*X_lu[1, 0]* \
                X_l[2, 1]+np.exp(- t11-t25-t44-t62)*X_u[1, 1]*X_lu[0, 1]*X_l[2, 0]+np.exp(- t11-t25-t41-t59)*X_u[2, 1]* \
                X_lu[0, 2]*X_l[2, 0]+np.exp(- t5-t25-t46-t61)*X_u[0, 1]*X_lu[2, 0]*X_l[2, 2]+np.exp(- t8-t25-t44-t60)* \
                X_u[1, 1]*X_lu[1, 1]*X_l[2, 1]+np.exp(- t11-t25-t38-t55)*X_u[3, 1]*X_lu[0, 3]*X_l[2, 0]+np.exp(
        - t8-t25-t41-t56)*X_u[2, 1]*X_lu[1, 2]*X_l[2, 1]+np.exp(- t5-t25-t44-t57)*X_u[1, 1]*X_lu[2, 1]*X_l[2, 2]+np.exp(
        - t3-t25-t46-t58)*X_u[0, 1]*X_lu[3, 0]*X_l[2, 3]+np.exp(- t8-t25-t38-t52)*X_u[3, 1]*X_lu[1, 3]*X_l[2, 1]+np.exp(
        - t5-t25-t41-t53)*X_u[2, 1]*X_lu[2, 2]*X_l[2, 2]+np.exp(- t3-t25-t44-t54)*X_u[1, 1]*X_lu[3, 1]*X_l[2, 3]+np.exp(
        - t5-t25-t38-t50)*X_u[3, 1]*X_lu[2, 3]*X_l[2, 2]+np.exp(- t3-t25-t41-t51)*X_u[2, 1]*X_lu[3, 2]*X_l[2, 3]+np.exp(
        - t3-t25-t38-t49)*X_u[3, 1]*X_lu[3, 3]*X_l[2, 3]
    Xmean[2, 2]=np.exp(- t11-t21-t43-t64)*X_u[0, 2]*X_lu[0, 0]*X_l[2, 0]+np.exp(- t11-t21-t40-t62)*X_u[1, 2]*X_lu[0, 1]* \
                X_l[2, 0]+np.exp(- t8-t21-t43-t63)*X_u[0, 2]*X_lu[1, 0]*X_l[2, 1]+np.exp(- t11-t21-t37-t59)*X_u[2, 2]* \
                X_lu[0, 2]*X_l[2, 0]+np.exp(- t8-t21-t40-t60)*X_u[1, 2]*X_lu[1, 1]*X_l[2, 1]+np.exp(- t5-t21-t43-t61)* \
                X_u[0, 2]*X_lu[2, 0]*X_l[2, 2]+np.exp(- t8-t21-t37-t56)*X_u[2, 2]*X_lu[1, 2]*X_l[2, 1]+np.exp(
        - t11-t21-t35-t55)*X_u[3, 2]*X_lu[0, 3]*X_l[2, 0]+np.exp(- t5-t21-t40-t57)*X_u[1, 2]*X_lu[2, 1]*X_l[
                    2, 2]+np.exp(- t3-t21-t43-t58)*X_u[0, 2]*X_lu[3, 0]*X_l[2, 3]+np.exp(- t5-t21-t37-t53)*X_u[2, 2]* \
                X_lu[2, 2]*X_l[2, 2]+np.exp(- t8-t21-t35-t52)*X_u[3, 2]*X_lu[1, 3]*X_l[2, 1]+np.exp(- t3-t21-t40-t54)* \
                X_u[1, 2]*X_lu[3, 1]*X_l[2, 3]+np.exp(- t5-t21-t35-t50)*X_u[3, 2]*X_lu[2, 3]*X_l[2, 2]+np.exp(
        - t3-t21-t37-t51)*X_u[2, 2]*X_lu[3, 2]*X_l[2, 3]+np.exp(- t3-t21-t35-t49)*X_u[3, 2]*X_lu[3, 3]*X_l[2, 3]
    Xmean[2, 3]=np.exp(- t11-t18-t39-t64)*X_u[0, 3]*X_lu[0, 0]*X_l[2, 0]+np.exp(- t11-t18-t36-t62)*X_u[1, 3]*X_lu[0, 1]* \
                X_l[2, 0]+np.exp(- t8-t18-t39-t63)*X_u[0, 3]*X_lu[1, 0]*X_l[2, 1]+np.exp(- t8-t18-t36-t60)*X_u[1, 3]* \
                X_lu[1, 1]*X_l[2, 1]+np.exp(- t11-t18-t34-t59)*X_u[2, 3]*X_lu[0, 2]*X_l[2, 0]+np.exp(- t5-t18-t39-t61)* \
                X_u[0, 3]*X_lu[2, 0]*X_l[2, 2]+np.exp(- t5-t18-t36-t57)*X_u[1, 3]*X_lu[2, 1]*X_l[2, 2]+np.exp(
        - t8-t18-t34-t56)*X_u[2, 3]*X_lu[1, 2]*X_l[2, 1]+np.exp(- t11-t18-t33-t55)*X_u[3, 3]*X_lu[0, 3]*X_l[
                    2, 0]+np.exp(- t3-t18-t39-t58)*X_u[0, 3]*X_lu[3, 0]*X_l[2, 3]+np.exp(- t5-t18-t34-t53)*X_u[2, 3]* \
                X_lu[2, 2]*X_l[2, 2]+np.exp(- t3-t18-t36-t54)*X_u[1, 3]*X_lu[3, 1]*X_l[2, 3]+np.exp(- t8-t18-t33-t52)* \
                X_u[3, 3]*X_lu[1, 3]*X_l[2, 1]+np.exp(- t3-t18-t34-t51)*X_u[2, 3]*X_lu[3, 2]*X_l[2, 3]+np.exp(
        - t5-t18-t33-t50)*X_u[3, 3]*X_lu[2, 3]*X_l[2, 2]+np.exp(- t3-t18-t33-t49)*X_u[3, 3]*X_lu[3, 3]*X_l[2, 3]
    Xmean[3, 0]=np.exp(- t7-t26-t48-t64)*X_u[0, 0]*X_lu[0, 0]*X_l[3, 0]+np.exp(- t4-t26-t48-t63)*X_u[0, 0]*X_lu[1, 0]* \
                X_l[3, 1]+np.exp(- t7-t26-t47-t62)*X_u[1, 0]*X_lu[0, 1]*X_l[3, 0]+np.exp(- t2-t26-t48-t61)*X_u[0, 0]* \
                X_lu[2, 0]*X_l[3, 2]+np.exp(- t4-t26-t47-t60)*X_u[1, 0]*X_lu[1, 1]*X_l[3, 1]+np.exp(- t7-t26-t45-t59)* \
                X_u[2, 0]*X_lu[0, 2]*X_l[3, 0]+np.exp(- t7-t26-t42-t55)*X_u[3, 0]*X_lu[0, 3]*X_l[3, 0]+np.exp(
        - t4-t26-t45-t56)*X_u[2, 0]*X_lu[1, 2]*X_l[3, 1]+np.exp(- t2-t26-t47-t57)*X_u[1, 0]*X_lu[2, 1]*X_l[3, 2]+np.exp(
        - t1-t26-t48-t58)*X_u[0, 0]*X_lu[3, 0]*X_l[3, 3]+np.exp(- t4-t26-t42-t52)*X_u[3, 0]*X_lu[1, 3]*X_l[3, 1]+np.exp(
        - t2-t26-t45-t53)*X_u[2, 0]*X_lu[2, 2]*X_l[3, 2]+np.exp(- t1-t26-t47-t54)*X_u[1, 0]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t2-t26-t42-t50)*X_u[3, 0]*X_lu[2, 3]*X_l[3, 2]+np.exp(- t1-t26-t45-t51)*X_u[2, 0]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t1-t26-t42-t49)*X_u[3, 0]*X_lu[3, 3]*X_l[3, 3]
    Xmean[3, 1]=np.exp(- t7-t22-t46-t64)*X_u[0, 1]*X_lu[0, 0]*X_l[3, 0]+np.exp(- t4-t22-t46-t63)*X_u[0, 1]*X_lu[1, 0]* \
                X_l[3, 1]+np.exp(- t7-t22-t44-t62)*X_u[1, 1]*X_lu[0, 1]*X_l[3, 0]+np.exp(- t7-t22-t41-t59)*X_u[2, 1]* \
                X_lu[0, 2]*X_l[3, 0]+np.exp(- t4-t22-t44-t60)*X_u[1, 1]*X_lu[1, 1]*X_l[3, 1]+np.exp(- t2-t22-t46-t61)* \
                X_u[0, 1]*X_lu[2, 0]*X_l[3, 2]+np.exp(- t7-t22-t38-t55)*X_u[3, 1]*X_lu[0, 3]*X_l[3, 0]+np.exp(
        - t4-t22-t41-t56)*X_u[2, 1]*X_lu[1, 2]*X_l[3, 1]+np.exp(- t2-t22-t44-t57)*X_u[1, 1]*X_lu[2, 1]*X_l[3, 2]+np.exp(
        - t4-t22-t38-t52)*X_u[3, 1]*X_lu[1, 3]*X_l[3, 1]+np.exp(- t1-t22-t46-t58)*X_u[0, 1]*X_lu[3, 0]*X_l[3, 3]+np.exp(
        - t2-t22-t41-t53)*X_u[2, 1]*X_lu[2, 2]*X_l[3, 2]+np.exp(- t1-t22-t44-t54)*X_u[1, 1]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t2-t22-t38-t50)*X_u[3, 1]*X_lu[2, 3]*X_l[3, 2]+np.exp(- t1-t22-t41-t51)*X_u[2, 1]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t1-t22-t38-t49)*X_u[3, 1]*X_lu[3, 3]*X_l[3, 3]
    Xmean[3, 2]=np.exp(- t7-t19-t43-t64)*X_u[0, 2]*X_lu[0, 0]*X_l[3, 0]+np.exp(- t7-t19-t40-t62)*X_u[1, 2]*X_lu[0, 1]* \
                X_l[3, 0]+np.exp(- t4-t19-t43-t63)*X_u[0, 2]*X_lu[1, 0]*X_l[3, 1]+np.exp(- t7-t19-t37-t59)*X_u[2, 2]* \
                X_lu[0, 2]*X_l[3, 0]+np.exp(- t4-t19-t40-t60)*X_u[1, 2]*X_lu[1, 1]*X_l[3, 1]+np.exp(- t2-t19-t43-t61)* \
                X_u[0, 2]*X_lu[2, 0]*X_l[3, 2]+np.exp(- t4-t19-t37-t56)*X_u[2, 2]*X_lu[1, 2]*X_l[3, 1]+np.exp(
        - t7-t19-t35-t55)*X_u[3, 2]*X_lu[0, 3]*X_l[3, 0]+np.exp(- t2-t19-t40-t57)*X_u[1, 2]*X_lu[2, 1]*X_l[3, 2]+np.exp(
        - t4-t19-t35-t52)*X_u[3, 2]*X_lu[1, 3]*X_l[3, 1]+np.exp(- t1-t19-t43-t58)*X_u[0, 2]*X_lu[3, 0]*X_l[3, 3]+np.exp(
        - t2-t19-t37-t53)*X_u[2, 2]*X_lu[2, 2]*X_l[3, 2]+np.exp(- t1-t19-t40-t54)*X_u[1, 2]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t2-t19-t35-t50)*X_u[3, 2]*X_lu[2, 3]*X_l[3, 2]+np.exp(- t1-t19-t37-t51)*X_u[2, 2]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t1-t19-t35-t49)*X_u[3, 2]*X_lu[3, 3]*X_l[3, 3]
    Xmean[3, 3]=np.exp(- t7-t17-t39-t64)*X_u[0, 3]*X_lu[0, 0]*X_l[3, 0]+np.exp(- t7-t17-t36-t62)*X_u[1, 3]*X_lu[0, 1]* \
                X_l[3, 0]+np.exp(- t4-t17-t39-t63)*X_u[0, 3]*X_lu[1, 0]*X_l[3, 1]+np.exp(- t4-t17-t36-t60)*X_u[1, 3]* \
                X_lu[1, 1]*X_l[3, 1]+np.exp(- t7-t17-t34-t59)*X_u[2, 3]*X_lu[0, 2]*X_l[3, 0]+np.exp(- t2-t17-t39-t61)* \
                X_u[0, 3]*X_lu[2, 0]*X_l[3, 2]+np.exp(- t4-t17-t34-t56)*X_u[2, 3]*X_lu[1, 2]*X_l[3, 1]+np.exp(
        - t2-t17-t36-t57)*X_u[1, 3]*X_lu[2, 1]*X_l[3, 2]+np.exp(- t7-t17-t33-t55)*X_u[3, 3]*X_lu[0, 3]*X_l[3, 0]+np.exp(
        - t1-t17-t39-t58)*X_u[0, 3]*X_lu[3, 0]*X_l[3, 3]+np.exp(- t2-t17-t34-t53)*X_u[2, 3]*X_lu[2, 2]*X_l[3, 2]+np.exp(
        - t4-t17-t33-t52)*X_u[3, 3]*X_lu[1, 3]*X_l[3, 1]+np.exp(- t1-t17-t36-t54)*X_u[1, 3]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t2-t17-t33-t50)*X_u[3, 3]*X_lu[2, 3]*X_l[3, 2]+np.exp(- t1-t17-t34-t51)*X_u[2, 3]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t1-t17-t33-t49)*X_u[3, 3]*X_lu[3, 3]*X_l[3, 3]

    return Xmean

def calc_iso_Xmean(X_l, X_lu, X_u, k, k_l, k_u, dd_u, dd_l, sigma, sigma_l, sigma_u):
    '''Function to assemble Xmean used for the isotropic case, 2x2 matrices, for the xrmr module.'''

    Xmean=np.empty(X_l.shape, dtype=np.complex128)

    t1=(1./2.)*sigma_l[:-1]**2*(k[:, :-1]-k_l[:, :-1])**2
    t2=(1./2.)*sigma_l[:-1]**2*(k[:, :-1]+k_l[:, :-1])**2
    t3=(1./2.)*sigma[1:]**2*(k[:, 1:]-k[:, :-1])**2
    t4=(1./2.)*sigma[1:]**2*(k[:, 1:]+k[:, :-1])**2
    t5=(1./2.)*sigma_u[1:]**2*(k[:, 1:]-k_u[:, 1:])**2
    t6=(1./2.)*sigma_u[1:]**2*(k[:, 1:]+k_u[:, 1:])**2

    Xmean[0, 0]=np.exp(- t1-t3-t5-dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[0, 0]*X_l[
        0, 0]+np.exp(- t1-t3-t6+dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[0, 1]*X_l[
                    0, 0]+np.exp(- t2-t3-t5-dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[1, 0]* \
                X_l[0, 1]+np.exp(- t2-t3-t6+dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[1, 1]* \
                X_l[0, 1]
    Xmean[0, 1]=np.exp(- t1-t4-t6-dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[0, 0]*X_l[
        0, 0]+np.exp(- t1-t4-t5+dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[0, 1]*X_l[
                    0, 0]+np.exp(- t2-t4-t6-dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[1, 0]* \
                X_l[0, 1]+np.exp(- t2-t4-t5+dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[1, 1]* \
                X_l[0, 1]
    Xmean[1, 0]=np.exp(- t2-t4-t5-dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[0, 0]*X_l[
        1, 0]+np.exp(- t1-t4-t5-dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[1, 0]*X_l[
                    1, 1]+np.exp(- t2-t4-t6+dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[0, 1]* \
                X_l[1, 0]+np.exp(- t1-t4-t6+dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[1, 1]* \
                X_l[1, 1]
    Xmean[1, 1]=np.exp(- t2-t3-t6-dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[0, 0]*X_l[
        1, 0]+np.exp(- t1-t3-t6-dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[1, 0]*X_l[
                    1, 1]+np.exp(- t2-t3-t5+dd_u[1:]*k_u[:, 1:]*1.0J-dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[0, 1]* \
                X_l[1, 0]+np.exp(- t1-t3-t5+dd_u[1:]*k_u[:, 1:]*1.0J+dd_l[:-1]*k_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[1, 1]* \
                X_l[1, 1]

    return Xmean

def calc_neu_Xmean(X_l, X_lu, X_u, km, kp, km_l, kp_l, km_u, kp_u, dd_u, dd_l, sigma, sigma_l, sigma_u):
    '''Function to assemble Xmean used for neutron calcs, 4x4 matrices, for the mag_refl module.'''

    Xmean=np.empty(X_l.shape, dtype=np.complex128)

    t1=(1./2.)*sigma_l[:-1]**2*(kp[:, :-1]-kp_l[:, :-1])**2
    t2=(1./2.)*sigma_l[:-1]**2*(kp[:, :-1]+kp_l[:, :-1])**2
    t3=(1./2.)*sigma_l[:-1]**2*(kp[:, :-1]-km_l[:, :-1])**2
    t4=(1./2.)*sigma_l[:-1]**2*(km[:, :-1]-kp_l[:, :-1])**2
    t5=(1./2.)*sigma_l[:-1]**2*(kp[:, :-1]+km_l[:, :-1])**2
    t6=(1./2.)*sigma_l[:-1]**2*(km[:, :-1]+kp_l[:, :-1])**2
    t7=(1./2.)*sigma_l[:-1]**2*(km[:, :-1]-km_l[:, :-1])**2
    t8=(1./2.)*sigma_l[:-1]**2*(km[:, :-1]+km_l[:, :-1])**2
    t9=(1./2.)*sigma[1:]**2*(kp[:, 1:]-kp[:, :-1])**2
    t10=(1./2.)*sigma[1:]**2*(kp[:, 1:]+kp[:, :-1])**2
    t11=(1./2.)*sigma[1:]**2*(kp[:, 1:]-km[:, :-1])**2
    t12=(1./2.)*sigma[1:]**2*(km[:, 1:]-kp[:, :-1])**2
    t13=(1./2.)*sigma[1:]**2*(kp[:, 1:]+km[:, :-1])**2
    t14=(1./2.)*sigma[1:]**2*(km[:, 1:]+kp[:, :-1])**2
    t15=(1./2.)*sigma[1:]**2*(km[:, 1:]-km[:, :-1])**2
    t16=(1./2.)*sigma[1:]**2*(km[:, 1:]+km[:, :-1])**2
    t17=(1./2.)*sigma_u[1:]**2*(kp[:, 1:]-kp_u[:, 1:])**2
    t18=(1./2.)*sigma_u[1:]**2*(kp[:, 1:]+kp_u[:, 1:])**2
    t19=(1./2.)*sigma_u[1:]**2*(kp[:, 1:]-km_u[:, 1:])**2
    t20=(1./2.)*sigma_u[1:]**2*(km[:, 1:]-kp_u[:, 1:])**2
    t21=(1./2.)*sigma_u[1:]**2*(kp[:, 1:]+km_u[:, 1:])**2
    t22=(1./2.)*sigma_u[1:]**2*(km[:, 1:]+kp_u[:, 1:])**2
    t23=(1./2.)*sigma_u[1:]**2*(km[:, 1:]-km_u[:, 1:])**2
    t24=(1./2.)*sigma_u[1:]**2*(km[:, 1:]+km_u[:, 1:])**2

    Xmean[0, 0]=np.exp(- t1-t9-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[0, 0]*X_l[
        0, 0]+np.exp(- t1-t9-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[0, 1]*X_l[
                    0, 0]+np.exp(- t2-t9-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[
                    1, 0]*X_l[0, 1]+np.exp(- t1-t9-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 0]* \
                X_lu[0, 2]*X_l[0, 0]+np.exp(- t3-t9-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[
                    0, 0]*X_lu[2, 0]*X_l[0, 2]+np.exp(
        - t2-t9-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[1, 1]*X_l[0, 1]+np.exp(
        - t2-t9-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[1, 2]*X_l[0, 1]+np.exp(
        - t3-t9-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[2, 1]*X_l[0, 2]+np.exp(
        - t1-t9-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[0, 3]*X_l[0, 0]+np.exp(
        - t5-t9-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[3, 0]*X_l[0, 3]+np.exp(
        - t3-t9-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[2, 2]*X_l[0, 2]+np.exp(
        - t2-t9-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[1, 3]*X_l[0, 1]+np.exp(
        - t5-t9-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[3, 1]*X_l[0, 3]+np.exp(
        - t3-t9-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[2, 3]*X_l[0, 2]+np.exp(
        - t5-t9-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[3, 2]*X_l[0, 3]+np.exp(
        - t5-t9-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[3, 3]*X_l[0, 3]
    Xmean[0, 1]=np.exp(- t1-t10-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[0, 0]*X_l[
        0, 0]+np.exp(- t1-t10-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[0, 1]*X_l[
                    0, 0]+np.exp(- t2-t10-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[
                    1, 0]*X_l[0, 1]+np.exp(- t3-t10-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[
                    0, 1]*X_lu[2, 0]*X_l[0, 2]+np.exp(
        - t1-t10-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[0, 2]*X_l[0, 0]+np.exp(
        - t2-t10-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[1, 1]*X_l[0, 1]+np.exp(
        - t1-t10-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[0, 3]*X_l[0, 0]+np.exp(
        - t3-t10-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[2, 1]*X_l[0, 2]+np.exp(
        - t2-t10-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[1, 2]*X_l[0, 1]+np.exp(
        - t5-t10-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[3, 0]*X_l[0, 3]+np.exp(
        - t2-t10-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[1, 3]*X_l[0, 1]+np.exp(
        - t3-t10-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[2, 2]*X_l[0, 2]+np.exp(
        - t5-t10-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[3, 1]*X_l[0, 3]+np.exp(
        - t3-t10-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[2, 3]*X_l[0, 2]+np.exp(
        - t5-t10-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[3, 2]*X_l[0, 3]+np.exp(
        - t5-t10-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[3, 3]*X_l[0, 3]
    Xmean[0, 2]=np.exp(- t1-t12-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[0, 0]*X_l[
        0, 0]+np.exp(- t2-t12-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[1, 0]*X_l[
                    0, 1]+np.exp(- t1-t12-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[
                    0, 1]*X_l[0, 0]+np.exp(- t3-t12-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[
                    0, 2]*X_lu[2, 0]*X_l[0, 2]+np.exp(
        - t1-t12-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[0, 2]*X_l[0, 0]+np.exp(
        - t2-t12-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[1, 1]*X_l[0, 1]+np.exp(
        - t1-t12-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[0, 3]*X_l[0, 0]+np.exp(
        - t2-t12-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[1, 2]*X_l[0, 1]+np.exp(
        - t3-t12-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[2, 1]*X_l[0, 2]+np.exp(
        - t5-t12-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[3, 0]*X_l[0, 3]+np.exp(
        - t3-t12-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[2, 2]*X_l[0, 2]+np.exp(
        - t2-t12-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[1, 3]*X_l[0, 1]+np.exp(
        - t5-t12-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[3, 1]*X_l[0, 3]+np.exp(
        - t3-t12-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[2, 3]*X_l[0, 2]+np.exp(
        - t5-t12-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[3, 2]*X_l[0, 3]+np.exp(
        - t5-t12-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[3, 3]*X_l[0, 3]
    Xmean[0, 3]=np.exp(- t1-t14-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[0, 0]*X_l[
        0, 0]+np.exp(- t1-t14-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[0, 1]*X_l[
                    0, 0]+np.exp(- t2-t14-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[
                    1, 0]*X_l[0, 1]+np.exp(- t1-t14-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 3]*X_lu[0, 2]*X_l[0, 0]+np.exp(
        - t2-t14-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[1, 1]*X_l[0, 1]+np.exp(
        - t3-t14-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[2, 0]*X_l[0, 2]+np.exp(
        - t3-t14-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[2, 1]*X_l[0, 2]+np.exp(
        - t1-t14-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[0, 3]*X_l[0, 0]+np.exp(
        - t2-t14-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[1, 2]*X_l[0, 1]+np.exp(
        - t5-t14-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[3, 0]*X_l[0, 3]+np.exp(
        - t3-t14-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[2, 2]*X_l[0, 2]+np.exp(
        - t2-t14-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[1, 3]*X_l[0, 1]+np.exp(
        - t5-t14-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[3, 1]*X_l[0, 3]+np.exp(
        - t3-t14-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[2, 3]*X_l[0, 2]+np.exp(
        - t5-t14-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[3, 2]*X_l[0, 3]+np.exp(
        - t5-t14-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[3, 3]*X_l[0, 3]
    Xmean[1, 0]=np.exp(- t2-t10-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[0, 0]*X_l[
        1, 0]+np.exp(- t1-t10-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[1, 0]*X_l[
                    1, 1]+np.exp(- t2-t10-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[
                    0, 1]*X_l[1, 0]+np.exp(- t2-t10-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 0]*X_lu[0, 2]*X_l[1, 0]+np.exp(
        - t1-t10-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[1, 1]*X_l[1, 1]+np.exp(
        - t5-t10-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[2, 0]*X_l[1, 2]+np.exp(
        - t1-t10-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[1, 2]*X_l[1, 1]+np.exp(
        - t3-t10-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[3, 0]*X_l[1, 3]+np.exp(
        - t2-t10-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[0, 3]*X_l[1, 0]+np.exp(
        - t5-t10-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[2, 1]*X_l[1, 2]+np.exp(
        - t3-t10-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[3, 1]*X_l[1, 3]+np.exp(
        - t5-t10-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[2, 2]*X_l[1, 2]+np.exp(
        - t1-t10-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[1, 3]*X_l[1, 1]+np.exp(
        - t3-t10-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[3, 2]*X_l[1, 3]+np.exp(
        - t5-t10-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[2, 3]*X_l[1, 2]+np.exp(
        - t3-t10-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[3, 3]*X_l[1, 3]
    Xmean[1, 1]=np.exp(- t2-t9-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[0, 0]*X_l[
        1, 0]+np.exp(- t1-t9-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[1, 0]*X_l[
                    1, 1]+np.exp(- t2-t9-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[
                    0, 1]*X_l[1, 0]+np.exp(- t1-t9-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 1]* \
                X_lu[1, 1]*X_l[1, 1]+np.exp(- t2-t9-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 1]*X_lu[0, 2]*X_l[1, 0]+np.exp(
        - t5-t9-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[2, 0]*X_l[1, 2]+np.exp(
        - t2-t9-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[0, 3]*X_l[1, 0]+np.exp(
        - t3-t9-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[3, 0]*X_l[1, 3]+np.exp(
        - t1-t9-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[1, 2]*X_l[1, 1]+np.exp(
        - t5-t9-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[2, 1]*X_l[1, 2]+np.exp(
        - t1-t9-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[1, 3]*X_l[1, 1]+np.exp(
        - t3-t9-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[3, 1]*X_l[1, 3]+np.exp(
        - t5-t9-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[2, 2]*X_l[1, 2]+np.exp(
        - t3-t9-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[3, 2]*X_l[1, 3]+np.exp(
        - t5-t9-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[2, 3]*X_l[1, 2]+np.exp(
        - t3-t9-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[3, 3]*X_l[1, 3]
    Xmean[1, 2]=np.exp(- t2-t14-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[0, 0]*X_l[
        1, 0]+np.exp(- t1-t14-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[1, 0]*X_l[
                    1, 1]+np.exp(- t2-t14-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[
                    0, 1]*X_l[1, 0]+np.exp(- t2-t14-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 2]*X_lu[0, 2]*X_l[1, 0]+np.exp(
        - t5-t14-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[2, 0]*X_l[1, 2]+np.exp(
        - t1-t14-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[1, 1]*X_l[1, 1]+np.exp(
        - t3-t14-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[3, 0]*X_l[1, 3]+np.exp(
        - t1-t14-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[1, 2]*X_l[1, 1]+np.exp(
        - t2-t14-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[0, 3]*X_l[1, 0]+np.exp(
        - t5-t14-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[2, 1]*X_l[1, 2]+np.exp(
        - t1-t14-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[1, 3]*X_l[1, 1]+np.exp(
        - t3-t14-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[3, 1]*X_l[1, 3]+np.exp(
        - t5-t14-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[2, 2]*X_l[1, 2]+np.exp(
        - t3-t14-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[3, 2]*X_l[1, 3]+np.exp(
        - t5-t14-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[2, 3]*X_l[1, 2]+np.exp(
        - t3-t14-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[3, 3]*X_l[1, 3]
    Xmean[1, 3]=np.exp(- t2-t12-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[0, 0]*X_l[
        1, 0]+np.exp(- t2-t12-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[0, 1]*X_l[
                    1, 0]+np.exp(- t1-t12-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[
                    1, 0]*X_l[1, 1]+np.exp(- t1-t12-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    1, 3]*X_lu[1, 1]*X_l[1, 1]+np.exp(
        - t2-t12-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[0, 2]*X_l[1, 0]+np.exp(
        - t5-t12-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[2, 0]*X_l[1, 2]+np.exp(
        - t1-t12-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[1, 2]*X_l[1, 1]+np.exp(
        - t2-t12-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[0, 3]*X_l[1, 0]+np.exp(
        - t3-t12-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[3, 0]*X_l[1, 3]+np.exp(
        - t5-t12-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[2, 1]*X_l[1, 2]+np.exp(
        - t3-t12-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[3, 1]*X_l[1, 3]+np.exp(
        - t1-t12-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[1, 3]*X_l[1, 1]+np.exp(
        - t5-t12-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[2, 2]*X_l[1, 2]+np.exp(
        - t3-t12-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[3, 2]*X_l[1, 3]+np.exp(
        - t5-t12-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[2, 3]*X_l[1, 2]+np.exp(
        - t3-t12-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[3, 3]*X_l[1, 3]
    Xmean[2, 0]=np.exp(- t4-t11-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[0, 0]*X_l[
        2, 0]+np.exp(- t4-t11-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[0, 1]*X_l[
                    2, 0]+np.exp(- t6-t11-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[
                    1, 0]*X_l[2, 1]+np.exp(- t4-t11-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 0]*X_lu[0, 2]*X_l[2, 0]+np.exp(
        - t7-t11-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[2, 0]*X_l[2, 2]+np.exp(
        - t6-t11-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[1, 1]*X_l[2, 1]+np.exp(
        - t4-t11-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[0, 3]*X_l[2, 0]+np.exp(
        - t6-t11-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[1, 2]*X_l[2, 1]+np.exp(
        - t7-t11-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[2, 1]*X_l[2, 2]+np.exp(
        - t8-t11-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[3, 0]*X_l[2, 3]+np.exp(
        - t7-t11-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[2, 2]*X_l[2, 2]+np.exp(
        - t8-t11-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[3, 1]*X_l[2, 3]+np.exp(
        - t6-t11-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[1, 3]*X_l[2, 1]+np.exp(
        - t8-t11-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[3, 2]*X_l[2, 3]+np.exp(
        - t7-t11-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[2, 3]*X_l[2, 2]+np.exp(
        - t8-t11-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[3, 3]*X_l[2, 3]
    Xmean[2, 1]=np.exp(- t4-t13-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[0, 0]*X_l[
        2, 0]+np.exp(- t4-t13-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[0, 1]*X_l[
                    2, 0]+np.exp(- t6-t13-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[
                    1, 0]*X_l[2, 1]+np.exp(- t4-t13-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 1]*X_lu[0, 2]*X_l[2, 0]+np.exp(
        - t7-t13-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[2, 0]*X_l[2, 2]+np.exp(
        - t6-t13-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[1, 1]*X_l[2, 1]+np.exp(
        - t4-t13-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[0, 3]*X_l[2, 0]+np.exp(
        - t7-t13-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[2, 1]*X_l[2, 2]+np.exp(
        - t8-t13-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[3, 0]*X_l[2, 3]+np.exp(
        - t6-t13-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[1, 2]*X_l[2, 1]+np.exp(
        - t6-t13-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[1, 3]*X_l[2, 1]+np.exp(
        - t7-t13-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[2, 2]*X_l[2, 2]+np.exp(
        - t8-t13-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[3, 1]*X_l[2, 3]+np.exp(
        - t7-t13-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[2, 3]*X_l[2, 2]+np.exp(
        - t8-t13-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[3, 2]*X_l[2, 3]+np.exp(
        - t8-t13-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[3, 3]*X_l[2, 3]
    Xmean[2, 2]=np.exp(- t4-t15-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[0, 0]*X_l[
        2, 0]+np.exp(- t4-t15-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[0, 1]*X_l[
                    2, 0]+np.exp(- t6-t15-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[
                    1, 0]*X_l[2, 1]+np.exp(- t4-t15-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 2]*X_lu[0, 2]*X_l[2, 0]+np.exp(
        - t7-t15-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[2, 0]*X_l[2, 2]+np.exp(
        - t6-t15-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[1, 1]*X_l[2, 1]+np.exp(
        - t4-t15-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[0, 3]*X_l[2, 0]+np.exp(
        - t8-t15-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[3, 0]*X_l[2, 3]+np.exp(
        - t6-t15-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[1, 2]*X_l[2, 1]+np.exp(
        - t7-t15-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[2, 1]*X_l[2, 2]+np.exp(
        - t7-t15-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[2, 2]*X_l[2, 2]+np.exp(
        - t6-t15-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[1, 3]*X_l[2, 1]+np.exp(
        - t8-t15-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[3, 1]*X_l[2, 3]+np.exp(
        - t7-t15-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[2, 3]*X_l[2, 2]+np.exp(
        - t8-t15-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[3, 2]*X_l[2, 3]+np.exp(
        - t8-t15-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[3, 3]*X_l[2, 3]
    Xmean[2, 3]=np.exp(- t4-t16-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[0, 0]*X_l[
        2, 0]+np.exp(- t4-t16-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[0, 1]*X_l[
                    2, 0]+np.exp(- t6-t16-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[
                    1, 0]*X_l[2, 1]+np.exp(- t4-t16-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    2, 3]*X_lu[0, 2]*X_l[2, 0]+np.exp(
        - t6-t16-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[1, 1]*X_l[2, 1]+np.exp(
        - t7-t16-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[2, 0]*X_l[2, 2]+np.exp(
        - t4-t16-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[0, 3]*X_l[2, 0]+np.exp(
        - t7-t16-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[2, 1]*X_l[2, 2]+np.exp(
        - t6-t16-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[1, 2]*X_l[2, 1]+np.exp(
        - t8-t16-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[3, 0]*X_l[2, 3]+np.exp(
        - t7-t16-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[2, 2]*X_l[2, 2]+np.exp(
        - t8-t16-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[3, 1]*X_l[2, 3]+np.exp(
        - t6-t16-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[1, 3]*X_l[2, 1]+np.exp(
        - t7-t16-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[2, 3]*X_l[2, 2]+np.exp(
        - t8-t16-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[3, 2]*X_l[2, 3]+np.exp(
        - t8-t16-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[3, 3]*X_l[2, 3]
    Xmean[3, 0]=np.exp(- t6-t13-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[0, 0]*X_l[
        3, 0]+np.exp(- t4-t13-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[1, 0]*X_l[
                    3, 1]+np.exp(- t6-t13-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[
                    0, 1]*X_l[3, 0]+np.exp(- t4-t13-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    1, 0]*X_lu[1, 1]*X_l[3, 1]+np.exp(
        - t6-t13-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[0, 2]*X_l[3, 0]+np.exp(
        - t8-t13-t17-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[2, 0]*X_l[3, 2]+np.exp(
        - t4-t13-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[1, 2]*X_l[3, 1]+np.exp(
        - t7-t13-t17-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 0]*X_lu[3, 0]*X_l[3, 3]+np.exp(
        - t8-t13-t18+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[2, 1]*X_l[3, 2]+np.exp(
        - t6-t13-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[0, 3]*X_l[3, 0]+np.exp(
        - t8-t13-t19-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[2, 2]*X_l[3, 2]+np.exp(
        - t4-t13-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[1, 3]*X_l[3, 1]+np.exp(
        - t7-t13-t18+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 0]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t7-t13-t19-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 0]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t8-t13-t21+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[2, 3]*X_l[3, 2]+np.exp(
        - t7-t13-t21+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 0]*X_lu[3, 3]*X_l[3, 3]
    Xmean[3, 1]=np.exp(- t6-t11-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[0, 0]*X_l[
        3, 0]+np.exp(- t4-t11-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[1, 0]*X_l[
                    3, 1]+np.exp(- t6-t11-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[
                    0, 1]*X_l[3, 0]+np.exp(- t4-t11-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    1, 1]*X_lu[1, 1]*X_l[3, 1]+np.exp(
        - t8-t11-t18-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[2, 0]*X_l[3, 2]+np.exp(
        - t6-t11-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[0, 2]*X_l[3, 0]+np.exp(
        - t4-t11-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[1, 2]*X_l[3, 1]+np.exp(
        - t6-t11-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[0, 3]*X_l[3, 0]+np.exp(
        - t7-t11-t18-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 1]*X_lu[3, 0]*X_l[3, 3]+np.exp(
        - t8-t11-t17+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[2, 1]*X_l[3, 2]+np.exp(
        - t4-t11-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[1, 3]*X_l[3, 1]+np.exp(
        - t7-t11-t17+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 1]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t8-t11-t21-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[2, 2]*X_l[3, 2]+np.exp(
        - t8-t11-t19+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[2, 3]*X_l[3, 2]+np.exp(
        - t7-t11-t21-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 1]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t7-t11-t19+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 1]*X_lu[3, 3]*X_l[3, 3]
    Xmean[3, 2]=np.exp(- t6-t16-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[0, 0]*X_l[
        3, 0]+np.exp(- t4-t16-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[1, 0]*X_l[
                    3, 1]+np.exp(- t6-t16-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[
                    0, 1]*X_l[3, 0]+np.exp(- t8-t16-t20-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[
                    0, 2]*X_lu[2, 0]*X_l[3, 2]+np.exp(
        - t4-t16-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[1, 1]*X_l[3, 1]+np.exp(
        - t6-t16-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[0, 2]*X_l[3, 0]+np.exp(
        - t4-t16-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[1, 2]*X_l[3, 1]+np.exp(
        - t7-t16-t20-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 2]*X_lu[3, 0]*X_l[3, 3]+np.exp(
        - t6-t16-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[0, 3]*X_l[3, 0]+np.exp(
        - t8-t16-t22+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[2, 1]*X_l[3, 2]+np.exp(
        - t4-t16-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[1, 3]*X_l[3, 1]+np.exp(
        - t8-t16-t23-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[2, 2]*X_l[3, 2]+np.exp(
        - t7-t16-t22+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 2]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t7-t16-t23-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 2]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t8-t16-t24+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[2, 3]*X_l[3, 2]+np.exp(
        - t7-t16-t24+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 2]*X_lu[3, 3]*X_l[3, 3]
    Xmean[3, 3]=np.exp(- t6-t15-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[0, 0]*X_l[
        3, 0]+np.exp(- t4-t15-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[1, 0]*X_l[
                    3, 1]+np.exp(- t6-t15-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[
                    0, 1]*X_l[3, 0]+np.exp(- t4-t15-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[
                    1, 3]*X_lu[1, 1]*X_l[3, 1]+np.exp(
        - t6-t15-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[0, 2]*X_l[3, 0]+np.exp(
        - t8-t15-t22-dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[2, 0]*X_l[3, 2]+np.exp(
        - t4-t15-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[1, 2]*X_l[3, 1]+np.exp(
        - t8-t15-t20+dd_u[1:]*kp_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[2, 1]*X_l[3, 2]+np.exp(
        - t6-t15-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[0, 3]*X_l[3, 0]+np.exp(
        - t7-t15-t22-dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[0, 3]*X_lu[3, 0]*X_l[3, 3]+np.exp(
        - t4-t15-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*kp_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[1, 3]*X_l[3, 1]+np.exp(
        - t7-t15-t20+dd_u[1:]*kp_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[1, 3]*X_lu[3, 1]*X_l[3, 3]+np.exp(
        - t8-t15-t24-dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[2, 2]*X_l[3, 2]+np.exp(
        - t7-t15-t24-dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[2, 3]*X_lu[3, 2]*X_l[3, 3]+np.exp(
        - t8-t15-t23+dd_u[1:]*km_u[:, 1:]*1.0J-dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[2, 3]*X_l[3, 2]+np.exp(
        - t7-t15-t23+dd_u[1:]*km_u[:, 1:]*1.0J+dd_l[:-1]*km_l[:, :-1]*1.0J)*X_u[3, 3]*X_lu[3, 3]*X_l[3, 3]

    return Xmean
