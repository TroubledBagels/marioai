local json = require("json")

local client = {}
client.client = nil

function client.connect(host)
    local socket = require("socket")
    local port = 8080
    client.client = socket.connect(host, port)
    client.client:settimeout(nil)
end

function client.close()
    if client ~= nil then
        client.client:send("CLOSE\n")
        client.client:close()
    end
    client.client = nil
end

function client.isConnected()
    return client.client ~= nil
end

function client.sendData(data)
    local send = json.encode(data)
    client.client:send(send .. "\n")
    return send
end

function receiveLine()
    local line = ""
    local data = nil
    local err = nil
    while data ~= "\n" do
        data, err = client.client:receive(1)

        if err ~= nil then
            print("Error: " .. err)
            client.close()
            return nil
        end

        if data ~= nil and data ~= "\n" then
            line = line .. data
        end
    end

    return line
end

function client.receiveButtons()
    local data = receiveLine()

    if #data ~= #button_names then
        client.close()
        print("Error: Received data is not the correct length")
        return nil
    end

    local jstate = {}
    for i=1, #button_names do
        local button = button_names[i]
        jstate[button] = string.sub(data, i, i) == "1"
        if string.sub(data, i, i) ~= "0" and string.sub(data, i, i) ~= "1" then
            client.close()
            print("Error: Received data is not binary")
            return nil
        end
    end


    return jstate
end

function client.receiveHeader()
    local data = receiveLine()
    local header = {}
    for i=1, #data do
        header[i] = receiveLine()
    end

    return header
end

return client