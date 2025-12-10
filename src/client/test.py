import minescript


p = minescript.player()
center = [int(p.position[0]), int(p.position[1]), int(p.position[2])]
voxels = []
radius = 1
    
for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                b_pos = (center[0]+dx, center[1]+dy, center[2]+dz)
                block = minescript.get_block(b_pos[0], b_pos[1], b_pos[2])
                minescript.echo(f"block data: {block}")