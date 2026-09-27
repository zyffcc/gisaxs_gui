"""Same conditional network layers, distance updated in radius-feasible coordinates."""
import tensorflow as tf
from conditional_resolution import ConditionalCorrector,project_globals

def logit(v):
    v=tf.clip_by_value(v,1e-5,1-1e-5);return tf.math.log(v)-tf.math.log1p(-v)

def distance_floor(radius_coordinate):
    radius=tf.exp(radius_coordinate*tf.math.log(100.))
    return tf.maximum(0.,tf.math.log(2*radius*1.001/3.)/tf.math.log(500./3.))

def feasible_parameters(start,delta):
    p=tf.sigmoid(logit(start)+delta)
    lo0=distance_floor(start[...,0]);u0=(start[...,4]-lo0)/(1-lo0)
    lo=distance_floor(p[...,0]);u=tf.sigmoid(logit(u0)+delta[...,4])
    distance=lo+(1-lo)*u
    return tf.concat([p[...,:4],distance[...,None],p[...,5:]],-1)

class FeasibleDistanceCorrector(ConditionalCorrector):
    def call(self,b,training=False):
        x=b['x']
        for layer in self.blocks:x=layer(x,training=training)
        active=tf.cast(b['types']>0,tf.float32);wp=tf.nn.softmax(b['weights']+(1-active)*-1e4)
        cm=tf.cast(b['condition_mask'],tf.float32);cv=tf.where(cm>0,b['condition_values'],tf.zeros_like(b['condition_values']))
        state=tf.concat([tf.reshape(tf.one_hot(b['types'],4)[:,:,1:],[-1,12]),tf.reshape(b['params'],[-1,24]),wp,
            b['globals'],tf.cast(b['d']>0,tf.float32),tf.cast(b['res'][:,None]>0,tf.float32),b['context']],-1)
        dx=self.final(self.hidden(tf.concat([self.flat(x),state,cm,cv],-1),training=training))
        p=feasible_parameters(b['params'],tf.reshape(dx[:,:24],[-1,4,6]))
        w=b['weights']+dx[:,24:28]
        g=project_globals(tf.sigmoid(logit(b['globals'])+dx[:,28:32]),b['condition_values'],cm)
        return p,w,g
