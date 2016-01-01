"use strict";

module.exports = function( done ) {

	return done( null, {

		state: {
			file: '.STATE.json'
		},

		ws: {
			port: 8765,
		},

		updateInterval: 1000,

		mqtt: {
			url: 'mqtt://localhost:1883/',
			username: 'apache',
			password: 'dBPg09K6U34m'
		},

		boilers: [
			{
				name: "Bernd der Braubottich"
			}, {
				name: "Kurt von Kessel"
			}
		]/*,

		fields: {

			jacket: {
				upper: {
					temp: {
						status: "f",
						nominal: "f"
					},
					power: {
						status: 'f'
					},
					heater: {
						status: 'b'
					}
				}
			}
		}
		*/
	} );
}
