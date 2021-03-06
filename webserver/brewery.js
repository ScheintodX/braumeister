"use strict";

var log = require( './logging.js' );
var HM = require( './helpers.js' ).message,
    JS = require( './helpers.js' ).json;;
var Dot = require( 'dot-object' );
var _ = require( 'underscore' );
var E = require( './E.js' );
var jdp = require( 'jsondiffpatch' );

module.exports = function( devices ) {

	function diff( orig, changed ) {

		var result = {};
		var keys = _.union( _.keys( orig ), _.keys( changed ) );

		_.each( keys, function( key ){

			if( !( key in orig ) ){
				result[ key ] = changed[ key ];
			} else if( orig[ key ] !== changed[ key ] ) {
				result[ key ] = changed[ key ];
			} else if( !( key in changed ) ){
				result[ key ] = null;
			}
		} );

		return result;
	}

	function doSupervised( f ) {

		var original = JSON.parse( self.asJson() );

		f();

		var changed = JSON.parse( self.asJson() );

		return jdp.diff( original, changed );
	}

	var self = {

		devices: devices,

		infrastructure: {},

		clone: function() {
			return JSON.parse( JSON.stringify( this ) );
		},

		asJson: function() {

			var devices = [];

			return JS.stringifyPublic( { devices: self.devices } );
		},

		// Direct access to sub fields via setByMqttMethod
		setByMqtt: function( topic, value ) {

			topic = HM.splitByMqtt( topic );

			if( topic[0] == "infrastructure" ){

				HM.setByParts( self, topic, value );

			} else if( topic[0] in self.devices ) {

				return doSupervised( function() {

					HM.setByMethod( self.devices, topic, value, 'setByMqtt' );

					self.watch();
				} );


			} else {
				log.warn( "Unknown topic: " + topic );
			}
		},

		// Direct access to sub fields via setByMqttMethod
		setByWeb: function( topic, value ) {

			topic = HM.splitByWeb( topic );

			console.log( topic );
			console.log( self.devices );

			if( topic[0] in self.devices ){

				console.log( "SET " + topic + " " + value, typeof value );

				log.info( "SET " + topic + " " + value, typeof value );

				return doSupervised( function() {

					HM.setByMethod( self.devices, topic, value, 'setByWeb' );

					self.watch();
				} );
			}
		},

		watch: function() {

			for( var device in self.devices ) {

				self.devices[ device ].watch( self );
			}
		},

		// Recursive access to fields via publish/emit
		publish: function( emit ) {

			for( var device in self.devices ) {

				var dev = self.devices[ device ];

				if( 'publish' in dev ) {

					dev.publish( function( topic, data ) {

						emit( device + '/' + topic, data );
					} );
				}
			}
		},

		// Recursive access to fields via subscribe/emit
		subscribe: function( emit ) {

			emit( 'infrastructure/#' );

			for( var device in self.devices ) {

				var dev = self.devices[ device ];

				if( 'subscribe' in dev ) {

					dev.subscribe( function( topic ) {

						emit( device + '/' + topic );
					} );
				}
			}
		}

	};

	return self;
}
