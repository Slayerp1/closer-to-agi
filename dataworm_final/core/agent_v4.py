"""
agent_v4.py — THE GOAL-FORMING WORM (clean rebuild)
Uses PlasticCuriosityEngine from brain_v3.
Same structure as agent_v3 but with plastic drives and energy.
"""
import numpy as np
from core.brain_v3 import PlasticCuriosityEngine, AdaptiveSensorArray, NerveRingV2

LEFT, FORWARD, RIGHT = 0, 1, 2
DIRECTIONS = [(-1,0),(0,1),(1,0),(0,-1)]

class DataWormV4:
    def __init__(self, start_x=None, start_y=None, env_width=40, env_height=40):
        self.x = start_x if start_x is not None else env_width // 2
        self.y = start_y if start_y is not None else env_height // 2
        self.direction = 0
        self.env_width = env_width
        self.env_height = env_height
        self.curiosity = PlasticCuriosityEngine()
        self.sensors = AdaptiveSensorArray()
        self.nerve_ring = NerveRingV2(n_inputs=12, n_hidden=30, n_outputs=3)
        self.memory_map = np.zeros((env_height, env_width))
        self.state = 'roaming'
        self.state_cooldown = 0
        self.reward_baseline = 0.2
        self.reward_ema_fast = 0.0
        self.reward_ema_slow = 0.0
        self.age = 0
        self.total_reward = 0.0
        self.danger_hits = 0
        self.data_found = 0.0
        self.novelty_seen = 0.0
        self.wall_bumps = 0
        self.unknown_encounters = 0
        self.adaptations = []
        self.recent_rewards = []
        self.position_history = []
        self.step_log = []

    def _sense_memory(self):
        def s(px,py):
            return float(self.memory_map[np.clip(py,0,self.env_height-1), np.clip(px,0,self.env_width-1)])
        return {'memory_L':s(self.x-1,self.y),'memory_R':s(self.x+1,self.y),
                'memory_F':s(self.x,self.y-1),'memory_here':s(self.x,self.y)}

    def _update_memory(self, reward):
        self.memory_map *= 0.997
        self.memory_map[self.y,self.x] = self.memory_map[self.y,self.x]*0.5 + reward*0.3

    def _update_state(self, reward):
        self.reward_ema_fast = 0.1*reward + 0.9*self.reward_ema_fast
        self.reward_ema_slow = 0.02*reward + 0.98*self.reward_ema_slow
        self.reward_baseline = 0.99*self.reward_baseline + 0.01*self.reward_ema_slow
        if self.curiosity.energy < self.curiosity.ENERGY_CRITICAL:
            self.state = 'roaming'; return
        rate = self.reward_ema_fast - self.reward_ema_slow
        if self.state_cooldown > 0: self.state_cooldown -= 1; return
        if self.state=='roaming' and self.reward_ema_fast > self.reward_baseline and rate > -0.01:
            self.state='dwelling'; self.state_cooldown=25
        elif self.state=='dwelling' and (rate < -0.02 or self.reward_ema_fast < self.reward_baseline*0.5):
            self.state='roaming'; self.state_cooldown=25

    def _danger_bias(self, sensors):
        fwd = max(sensors.get('danger_here',0), sensors.get('danger_F',0))
        thr = 0.7 if self.curiosity.energy > self.curiosity.ENERGY_CRITICAL else 0.9
        if fwd > thr: return 'emergency', 0.0, True
        asym = sensors.get('danger_L',0) - sensors.get('danger_R',0)
        if fwd > 0.3: return 'medium', asym*2.0, False
        if fwd > 0.1: return 'low', asym*0.8, False
        return 'none', 0.0, False

    def _try_move(self, env):
        bumps=0
        for attempt in range(4):
            dy,dx = DIRECTIONS[self.direction]
            nx,ny = np.clip(self.x+dx,0,env.width-1), np.clip(self.y+dy,0,env.height-1)
            if not env.is_wall(nx,ny): return int(nx),int(ny),bumps
            bumps+=1
            self.direction = (self.direction + (1 if attempt<2 else 2)) % 4
        return self.x,self.y,bumps

    def sense(self, env): return env.observe(self.x, self.y)
    def _rotate(self, a):
        if a==LEFT: self.direction=(self.direction-1)%4
        elif a==RIGHT: self.direction=(self.direction+1)%4

    def step(self, env):
        self.age += 1
        raw = self.sense(env)
        mem = self._sense_memory()
        svec, found_unk = self.sensors.process(raw)
        if found_unk and self.unknown_encounters==0:
            self.adaptations.append({'step':self.age,'type':'UNKNOWN_DETECTED','desc':'Growing connections.'})
        if found_unk: self.unknown_encounters += 1

        cout = self.curiosity.compute(raw)
        dlevel, dbias, reverse = self._danger_bias(raw)
        mprobs = self.nerve_ring.forward(svec)

        if len(self.position_history)>=20 and len(set(self.position_history[-20:]))<5 and self.age%5==0:
            self.direction = (self.direction+np.random.randint(1,4))%4

        reflex = False
        if reverse:
            self.direction=(self.direction+2)%4; action=FORWARD; reflex=True; self.danger_hits+=1
        else:
            b = mprobs.copy()
            db = cout['direction_bias']
            if self.state=='roaming':
                b[LEFT]+=max(0,-db)*0.4; b[RIGHT]+=max(0,db)*0.4; b[FORWARD]+=cout['forward_pull']*0.15
            else:
                b[FORWARD]+=0.3; b[LEFT]+=max(0,-db)*0.15; b[RIGHT]+=max(0,db)*0.15
            if dlevel=='medium':
                if dbias>0: b[RIGHT]+=abs(dbias)*0.5; b[FORWARD]*=0.3
                else: b[LEFT]+=abs(dbias)*0.5; b[FORWARD]*=0.3
            elif dlevel=='low':
                if dbias>0: b[RIGHT]+=abs(dbias)*0.2
                else: b[LEFT]+=abs(dbias)*0.2
            if self.state=='roaming' and self.reward_ema_fast < self.reward_baseline:
                ma=mem['memory_R']-mem['memory_L']
                b[RIGHT]+=max(0,ma)*0.3; b[LEFT]+=max(0,-ma)*0.3; b[FORWARD]+=mem['memory_F']*0.2
            b=np.clip(b,0.01,None); b/=b.sum()
            action=int(np.random.choice([0,1,2],p=b)); self._rotate(action)

        ox,oy=self.x,self.y
        self.x,self.y,wb=self._try_move(env); self.wall_bumps+=wb
        reward=env.step_into(self.x,self.y)
        if wb>0: reward-=0.1*wb
        self.total_reward+=reward
        self.data_found+=max(0,raw.get('richness_here',0))
        self.novelty_seen+=max(0,raw.get('novelty_here',0))
        self.recent_rewards.append(reward)
        if len(self.recent_rewards)>100: self.recent_rewards=self.recent_rewards[-100:]
        self.position_history.append((self.x,self.y))
        if len(self.position_history)>50: self.position_history=self.position_history[-50:]

        self._update_memory(reward)
        self.curiosity.update_energy(max(0,raw.get('richness_here',0)))
        self.curiosity.update_associations(raw, reward)
        self.curiosity.update_drives(raw,reward, raw.get('novelty_F',0)>0.5, raw.get('richness_F',0)>0.2)
        self._update_state(reward)
        wc,surprise = self.nerve_ring.hebbian_update(reward)

        d=self.curiosity.get_drive_stats()
        entry={'step':self.age,'x':self.x,'y':self.y,'action':['LEFT','FORWARD','RIGHT'][action],
               'direction':['N','E','S','W'][self.direction],'state':self.state,'reflex_fired':reflex,
               'danger_level':dlevel,'reward':round(reward,4),'surprise':round(surprise,4),
               'total_reward':round(self.total_reward,4),'curiosity_score':round(cout['curiosity_score'],4),
               'energy':d['energy'],'w_novelty':d['w_novelty'],'w_richness':d['w_richness'],
               'w_danger':d['w_danger'],'w_unknown':d['w_unknown'],'weight_change':round(wc,6),
               'n_inputs':self.nerve_ring.n_inputs,'found_unknown':found_unk,
               'data_found':round(self.data_found,3),'danger_hits':self.danger_hits,
               'wall_bumps':self.wall_bumps,'unknown_encounters':self.unknown_encounters}
        self.step_log.append(entry)
        return entry

    def get_stats(self):
        if not self.step_log: return {}
        return {'age':self.age,'total_reward':round(self.total_reward,3),
                'drives':self.curiosity.get_drive_stats(),'n_inputs':self.nerve_ring.n_inputs,
                'danger_hits':self.danger_hits,'wall_bumps':self.wall_bumps,
                'unknown_encounters':self.unknown_encounters,'adaptations':self.adaptations}
