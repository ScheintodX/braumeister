"use strict";

var E = require( './E.js' );
var Catch = require( './catch.js' );
var Dot = require( 'dot-object' ),
	Dash = new Dot( '/' );

var log = require( './logging.js' );
var Assert = require( './assert.js' );

var Scripts = require( './scripts.js' );


module.exports = function( config, hello, brewery ) {

	Assert.present( "config", config );
	Assert.present( "hello", hello );
	Assert.present( "brewery", brewery );

	var _mqtt, _web;

	function _msg( device, level, text ) {

		return _web( {
			message: {
				device: device, 
				level: level,
				messages: [
					{ level: level, text: text }
				]
			}
		} );
	}

	function _warn( device, text ) {
		return _msg( device, 'warn', text );
	}
	function _info( device, text ) {
		return _msg( 'info', text );
	}

	function sendStatusMqtt() {
		
		_mqtt( 'infrastructure/brewmaster/presence', "brewmaster" );

		brewery.watch();

		brewery.publish( _mqtt );
	}

	function sendStatusWeb() {

		_web( brewery.asJson() );
	}

	function gotWebDataSet( data ) {

		var val = data.value,
			topic = data.topic
			;

		var diff = brewery.setByWeb( topic, val );

		log.trace( JSON.stringify( diff ) );

		if( diff ) brewery.publish( _mqtt );
	}

	function gotWebDataLoadSave( data ) {

		Assert.present( 'data.device', data.device );
		Assert.present( 'data.topic', data.topic );

		var device = brewery.devices[ data.device ];

		if( ! device ) throw new Error( "No device found" );

		switch( data.topic ) {

			case "load": 

				log.trace( 'load', data.value.load );

				Scripts.load( data.value.load, function( err, Script, script ) {

					if( err ) return log.error( err );

					var TheScript = Script( script, device, config, {

						notify: function( device, what, message ){
							log.info( device.name, what, message );
						},

						time: config.script.time

					} );

					device.script = TheScript.hello;
					device._script = TheScript;

					sendStatusWeb();

					log.info( 'load done', data.value.load );
				} );

				break;

			case "save":

				log.trace( "SAVE", data.value.name );

				if( !device.script ) {
					_warn( data.device, 'No script available' );
					return;
				}

				var TheScript = device._script.parse( data.value );

				device.script = TheScript.hello;
				device._script = TheScript;

				var saveable = device._script.save();

				Scripts.save( data.value.name, saveable, function( err ) {

					if( err ) {
						_warn( data.device, err );
						return;
					}

					_info( "Saved" );
					log.trace( "SAVED", data.value.name );

					delete( hello.scripts ); // force reload

				} );

				break;

			case "set":

				log.trace( "SET" );

				if( !device.script ) {
					_warn( data.device, 'No script available' );
					return;
				}

				var TheScript = device._script.parse( data.value );

				device.script = TheScript.hello;
				device._script = TheScript;

				log.info( "SET done", device.script );

				break;

			default: 
				throw new Error( "Unknown action: " + data.topic );

		}

	}

	function gotWebDataRunStop( data ) {

		Assert.present( 'data.device', data.device );

		var device = brewery.devices[ data.device ];

		Assert.present( 'device', device );

		if( [ 'start', 'pause', 'resume', 'stop', 'next', 'prev' ].indexOf( data.topic ) >= 0 ){

			var script = device._script;

			Assert.present( 'script', script );

			script[ data.topic ]();

		} else {
			throw new Error( "Unknown action: " + data.topic );
		}
	}

	function isReady() {

		return _mqtt && _web;
	}

	var self = {

		gotWebData: Catch.fatal( "Ctrl/gotWeb", function( data ) {

			log.trace( "WebData", data );

			if( !isReady() ) {
				log.warn( "Premature web data: ", data );
			}

			switch( data.on ) {
				case "set": return gotWebDataSet( data ); break;
				case "loadsave": gotWebDataLoadSave( data ); break;
				case "runstop": gotWebDataRunStop( data ); break;
				default: throw new Error( "Unknown action: " + data.on );
			}

		} ),

		gotMqttData: Catch.fatal( "Ctrl/GotMqtt", function( topic, data ) {

			if( !isReady() ) {
				log.warn( "Premature mqtt data: ", topic, data );
			}

			var diff = brewery.setByMqtt( topic, data );

			if( diff ) {
				_web( { diff: diff } );
			}
		} ),

		onMqttMessage: function( mqtt ) {

			_mqtt = Catch.fatal( "Ctrl/Mqtt", mqtt );

			Assert.present( "config.updateIntervalMqtt", config.updateIntervalMqtt );

			setInterval( Catch.fatal( "Ctrl/SendStatusMqtt", sendStatusMqtt ), config.updateIntervalMqtt );
		},

		onWebMessage: function( web ) {

			_web = Catch.fatal( "Ctrl/WS", web );

			Assert.present( "config.updateIntervalWeb", config.updateIntervalWeb );

			setInterval( Catch.fatal( "Ctrl/SendStatusWeb", sendStatusWeb ), config.updateIntervalWeb );
		},

		run: function() {

			// Load script directory listing
			if( !( 'scripts' in hello ) ) {

				log.trace( "NOSCRIPT" );

				Scripts.list( function( err, data ) {

					if( err ) E.rr( err );

					if( err ) return log.error( err );

					hello.scripts = data;

					log.trace( "SEND", hello );

					_web( hello );

				} );
			}

			// Run available scripts from devices
			for( var key in brewery.devices ) {

				var device = brewery.devices[ key ];

				if( device._script ){
					device._script.run();
				}
			}
		},

		start: function() {

			setInterval( Catch.fatal( "Ctrl/Run", self.run ), config.updateIntervalCtrl );
		}
	};

	return self;
};

