SMBScreen = {}

BoxRadius = 6
InputSize = (BoxRadius*2+1)*(BoxRadius*2+1)

function getDetails(player)
    grounded = mainmemory.readbyte(0x001D)
    direction = mainmemory.readbyte(0x0033)
    hspeed = mainmemory.readbyte(0x0057)
    move_direction = mainmemory.readbyte(0x0045)
    px = mainmemory.readbyte(0x0086) + (mainmemory.readbyte(0x006D) * 256)
    py = mainmemory.readbyte(0x00CE)
    swimming = mainmemory.readbyte(0x0704)
    powerup_state = mainmemory.readbyte(0x0756)
    frame = mainmemory.readbyte(0x0009)
    state = mainmemory.readbyte(0x000E)
    fallen_off_screen = mainmemory.readbyte(0x00B5)
    dead = state == 6 or state == 0xB or fallen_off_screen > 1
end

function getSprites()
    local sprites = {}
    for slot=0,4 do
        local enemy = mainmemory.readbyte(0x00F+slot)
        if enemy ~= 0 then
            local ex = mainmemory.readbyte(0x006E+slot)*0x100 + mainmemory.readbyte(0x0087+slot)
            local ey = mainmemory.readbyte(0x00CF+slot) + 24
            sprites[#sprites+1] = {["x"]=ex, ["y"]=ey}
        end     
    end
    return sprites
end

function getTile(row, col) 
    local x = px + col + 8
    local y = py + row - 16
    local page = math.floor(x / 256)%2

    local subx = math.floor((x%256)/16)
    local suby = math.floor((y - 32)/16)
    local addr = 0x500 + page*13*16+suby*16+subx

    if suby >= 13 or suby < 0 then
        return 0
    end

    if mainmemory.readbyte(addr) == 0 then
        return 0
    else
        return 1
    end
end

function SMBScreen.getScreen(player)
    getDetails(player)
    local sprites = getSprites()
    local screen = {}

    for row=-BoxRadius*16, BoxRadius*16, 16 do
        for col=-BoxRadius*16, BoxRadius*16, 16 do
            if fallen_off_screen > 1 then
                screen[#screen+1] = 0
            end

            screen[#screen+1] = getTile(row, col)

            for i = 1, #sprites do
                distx = math.abs(sprites[i]["x"] - (px+col))
                disty = math.abs(sprites[i]["y"] - (py+row))
                if distx <= 8 and disty <= 8 then
                    screen[#screen] = -1
                end
            end
        end
    end
    local inputs = {grounded, direction, hspeed, move_direction, px, py, swimming, powerup_state, frame, state, fallen_off_screen, dead, screen}

    return inputs
end

return SMBScreen