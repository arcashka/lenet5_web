import http.server, ssl

server_address = ('', 8000)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)
httpd.socket = ssl.wrap_socket(httpd.socket,
                               server_side=True,
                               certfile='/home/arcashka/.ssh/self_signed_certs/cert.pem',
                               keyfile='/home/arcashka/.ssh/self_signed_certs/key.pem',
                               ssl_version=ssl.PROTOCOL_TLS)
httpd.serve_forever()
