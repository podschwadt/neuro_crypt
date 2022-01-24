import matplotlib.pyplot as plt

plt.ioff()
plt.ylim( (0,1) )
plt.plot( 1, .6, 'go', label='train' )
plt.plot( 2, .7, 'mo', label='val' )
plt.plot( 3, .8, 'co', label='test')
plt.title( 'training and eval' )
# plt.xlabel( '# Filters' )
plt.xticks(  [1,2,3], ['site1', 'site2', 'site3'] )
plt.ylabel( 'AUC' )
plt.grid(True)
plt.legend()
plt.show()