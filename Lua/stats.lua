Game = require "SMBScreen"

function text(x, y, str)
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

while true do
    screen = Game.getScreen(1)
    displayMap(screen[#screen])

    text(0, 0, "Grounded: " .. grounded)
    text(0, 15, "Direction: " .. direction)
    text(0, 30, "HSpeed: " .. hspeed)
    text(0, 45, "VSpeed: " .. vspeed)
    text(0, 60, "Move Direction: " .. move_direction)
    text(0, 75, "PX: " .. px)
    text(0, 90, "PY: " .. py)
    text(0, 105, "Swimming: " .. swimming)
    text(0, 120, "Powerup State: " .. powerup_state)
    text(0, 135, "Frame: " .. frame)
    text(0, 150, "State: " .. state)
    text(0, 165, "Fallen Off Screen: " .. fallen_off_screen)

    emu.frameadvance()
end