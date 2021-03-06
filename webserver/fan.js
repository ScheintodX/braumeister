"use strict";

require( './polyfill.js' );

var E = require( './E.js' );

var H = require( './helpers.js' );

var log = require( './logging.js' );

var InProxy = require( './sensor/in_proxy.js' ),
	OutProxy = require( './sensor/out_proxy.js' ),
	InOutProxy = require( './sensor/inout_proxy.js' ),
	Combined = require( './sensor/combined.js' )
	;

function createFan( index, config ) {

	var self = Object.assign( {

		name: config.name,
		index: index,
		conf: config,

	}, Combined( {

		indoor: Combined( {}, {
			temp: InProxy( {
				type: 'f',
				timeout: 2500
			} ),
			humidity_rel: InProxy( {
				type: 'f',
				timeout: 2500
			} ),
			humidity_abs: InProxy( {
				type: 'f',
				timeout: 2500
			} )
		} ),
		outdoor: Combined( {}, {
			temp: InProxy( {
				type: 'f',
				timeout: 2500
			} ),
			humidity_rel: InProxy( {
				type: 'f',
				timeout: 2500
			} ),
			humidity_abs: InProxy( {
				type: 'f',
				timeout: 2500
			} )
		} ),
		fan: Combined( {}, {
			mode: InOutProxy( {
					type: 's',
					timeout: 3000
				}, { set: 'auto' }
			),
			on: InProxy( {
				type: 'b',
				timeout: 3000
			}, {} )
		} )

	} ), {

		watch: function() {

		},

		power: function() {

			if( self.fan.on.status ) return self.conf.power;

			return 0;
		},

		powerLimit: function( limit ) {

			if( limit < self.conf.power ) self.fan.mode.set = 'off';
		}

	} );

	return self;
};

module.exports = {};

module.exports.create = createFan;
