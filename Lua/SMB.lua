local ltn12 = require("ltn12")
local json = require("json")
local Game = require("SMBScreen")
local client = require("client")
client.connect("localhost")

local function text(x, y, str)
    gui.text(x, y, str, "white", "black")
end

local function drawBox(x, y, width, height, color)
    gui.drawBox(x, y, x + width, y + height, color, color)
end

-- Map is 13x13 (assumed) - 1 indexed means weird maths
local function displayMap(map)
    posX = 150
    posY = 0
    boxSize = 4
    for i=1, BoxRadius*2+1 do
        for j=1, BoxRadius*2+1 do
            if map[(i-1) * 13 + j] == 1 then
                drawBox(posX + (j * boxSize), posY + (i * boxSize), boxSize, boxSize, "white")
            elseif map[(i-1) * 13 + j] == -1 then
                drawBox(posX + (j * boxSize), posY + (i * boxSize), boxSize, boxSize, "black")
            end
        end
    end
end

button_names = {
    "A", "B", "Left", "Right"
}

grounded = false -- 0x001D, 0 on ground, 1 in air from jump, 2 in air from falling, 3 in air from flagpole10 -
direction = 0 -- 0x0033, 0 not on screen, 1 left, 2 right
hspeed = 0 -- 0x0057, 0x00 not moving, 0xD8<0 moving left, 0x28>0 moving right 
vspeed = 0 
move_direction = 0 -- 0x0045, 1 right, 2 left
px = 0 -- 0x0086 (this screen, i.e. 256 unit chunk) + (0x006D * 256)
py = 0 -- 0x00CE screen pos - (multiply by 0x00B5 to get level y position)
swimming = 1 -- 0x0704, 0 swimming, 1 not swimming
powerup_state = 1 -- 0x0756, 0 small, 1 big, 2 fire
frame = 0 -- 0x0009, counts up for each frame active
state = 0 -- 0x000E, 0 leftmost, 1 climbing vine, 2 entering reverse-L pipe, 3 going down pipe, 4 autowalk, 5 autowalk, 6 dies, 7 entering area, 8 normal, 9 transforming small to large, A large to small, B dying, C transforming to fire
fallen_off_screen = 0 -- 0x00B5, 0 if on screen, >1 if fallen off screen
dead = false -- bool representing whether a player is dead or not - calculated with state and fallen_off_screen
localframe = 0

local jstate = joypad.get(1)

rightmost = 0
TimeoutConstant = 20

while (true) do
    if localframe % 4 == 0 then
        screen = Game.getScreen(1)
        screen[9] = localframe
        displayMap(screen[#screen])
        client.sendData(screen)
        jstate = client.receiveButtons()
    end

    grounded = screen[1]
    direction = screen[2]
    hspeed = screen[3]
    move_direction = screen[4]
    px = screen[5]
    py = screen[6]
    swimming = screen[7]
    powerup_state = screen[8]
    frame = screen[9]
    state = screen[10]
    fallen_off_screen = screen[11]
    dead = screen[12]

    text(0, 0, "Grounded: " .. grounded)
    text(0, 20, "Direction: " .. direction)
    text(0, 40, "Hspeed: " .. hspeed)
    text(0, 60, "Move Direction: " .. move_direction)
    text(0, 80, "X: " .. px)
    text(0, 100, "Y: " .. py)
    text(0, 120, "Swimming: " .. swimming)
    text(0, 140, "Power State: " .. powerup_state)
    text(0, 160, "Frame: " .. frame)
    text(0, 180, "Dead: " .. tostring(dead))
    text(0, 200, "State: " .. state)
    text(0, 220, "Fallen Off Screen: " .. fallen_off_screen)

    if px > rightmost then
        rightmost = px
        timeout = TimeoutConstant
    end

    timeout = timeout - 1

    currentJoypad = joypad.get(1)
    for i=1, #button_names do
        button = button_names[i]
        currentJoypad[button] = jstate[button]
    end
    joypad.set(currentJoypad, 1)
    localframe = localframe + 1

    emu.frameadvance()
end