% size of the sample that is plotted
N = 10000;

data = csvread('~/Seafile/My Library/basicmodule_prokofjevs/ds_uniform.csv');
% data = csvread('~/Seafile/My Library/basicmodule_prokofjevs/ds_gaussian.csv');

x = data(1:N,1);
y = data(1:N,2);
scatter(x,y)
